"""Capture and inference threads with a latest-frame handoff.

The camera and the CPU-bound inference run independently, so a slow frame never
stalls the reader and never queues up: the pipeline keeps only the newest frame
(`LatestValue`). Dropping frames is correct here: a stale frame showing a face
that already left the view is worse than a dropped one.

Nothing in this module touches Qt. The GUI reads `wait_output()` from a timer and
the CLI from a loop, so the same pipeline serves both.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

from config import AppConfig

from .camera import Camera, CameraError, Frame
from .face_recognizer import FrameRecognition
from .face_registration import SampleAttempt
from .metrics import RateCounter

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Mode(enum.StrEnum):
    """What the inference thread is doing."""

    IDLE = "idle"
    RECOGNIZE = "recognize"
    REGISTER = "register"


class FrameProcessor(Protocol):
    """Anything that turns a frame into a recognition result."""

    def process(self, image: np.ndarray, timestamp: float | None = None) -> FrameRecognition: ...


class RegistrationProcessor(Protocol):
    """Anything that turns a frame into an enrollment attempt."""

    def submit_frame(self, frame: np.ndarray, /) -> SampleAttempt: ...

    def reset(self) -> None: ...


class CaptureSource(Protocol):
    """A source of frames, camera or file."""

    def read(self) -> Frame | None: ...

    def describe(self) -> str: ...

    def close(self) -> None: ...


class LatestValue:
    """Single-slot handoff that always keeps the newest value."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: object | None = None
        self._version = 0
        self._closed = False

    def set(self, value: object) -> None:
        with self._condition:
            self._value = value
            self._version += 1
            self._condition.notify_all()

    @property
    def version(self) -> int:
        with self._condition:
            return self._version

    def peek(self) -> object | None:
        with self._condition:
            return self._value

    def get(self) -> object | None:
        with self._condition:
            return self._value

    def wait_newer(self, version: int, timeout: float) -> object | None:
        """Wait until a value newer than ``version`` arrives (or time out)."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._version <= version and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._value if self._version > version else None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class CameraSource:
    """Live camera as a capture source.

    ``Camera.read`` raises instead of returning nothing, because a missing frame is
    an error for a single reader. Here a failure is one dropped frame: the capture
    loop counts it and only gives up after a run of them.
    """

    def __init__(self, camera: Camera) -> None:
        self._camera = camera

    def read(self) -> Frame | None:
        try:
            return self._camera.read()
        except CameraError as exc:
            logger.debug("Camera read failed: %s", exc)
            return None

    def describe(self) -> str:
        return self._camera.describe()

    def close(self) -> None:
        self._camera.close()


class ImageSource:
    """A still image, optionally repeated, for tests and benchmarks."""

    def __init__(self, image: np.ndarray, label: str = "image", repeat: bool = True) -> None:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("ImageSource needs a non-empty image.")
        self._image = np.asarray(image)
        self._label = label
        self._repeat = repeat
        self._sequence = 0
        self._done = False
        self._closed = False

    def read(self) -> Frame | None:
        if self._closed or (self._done and not self._repeat):
            return None
        self._sequence += 1
        if not self._repeat:
            self._done = True
        return Frame(self._image.copy(), self._sequence)

    def describe(self) -> str:
        height, width = self._image.shape[:2]
        return f"{self._label} ({width}x{height})"

    def close(self) -> None:
        self._closed = True


class CaptureThread(threading.Thread):
    """Read frames from a source into a `LatestValue`."""

    def __init__(self, source: CaptureSource, sink: LatestValue, config: AppConfig) -> None:
        super().__init__(name="face-id-capture", daemon=True)
        self._source = source
        self._sink = sink
        self._config = config
        self._stop_event = threading.Event()
        self._rate = RateCounter()
        self._failures = 0
        self._error: str | None = None

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def fps(self) -> float:
        return self._rate.rate()

    def run(self) -> None:
        limit = self._config.camera.max_consecutive_failures
        while not self._stop_event.is_set():
            frame = self._source.read()
            if frame is None:
                self._failures += 1
                if self._failures >= limit:
                    self._error = (
                        f"The capture source stopped returning frames "
                        f"({self._failures} attempts)."
                    )
                    logger.error(self._error)
                    break
                time.sleep(0.02)
                continue
            self._failures = 0
            self._rate.tick()
            self._sink.set(frame)


@dataclass(frozen=True)
class PipelineOutput:
    """One processed frame, as produced by the inference thread."""

    frame: Frame
    mode: Mode
    recognition: FrameRecognition | None = None
    attempt: SampleAttempt | None = None
    inference_ms: float = 0.0


class InferenceThread(threading.Thread):
    """Run detection/recognition/enrollment on the newest frame."""

    def __init__(
        self,
        source: LatestValue,
        sink: LatestValue,
        config: AppConfig,
    ) -> None:
        super().__init__(name="face-id-inference", daemon=True)
        self._source = source
        self._sink = sink
        self._config = config
        self._stop_event = threading.Event()
        self._mode = Mode.IDLE
        self._mode_lock = threading.Lock()
        self._recognition: FrameProcessor | None = None
        self._registration: RegistrationProcessor | None = None
        self._rate = RateCounter()
        # The slot counts publishes; the camera numbers frames and skips some. Waiting
        # must use the slot counter, or a skipped number parks this thread far behind.
        self._last_sequence = 0
        self._dropped = 0
        self._error: str | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def set_mode(self, mode: Mode) -> None:
        with self._mode_lock:
            self._mode = mode

    @property
    def mode(self) -> Mode:
        with self._mode_lock:
            return self._mode

    def set_recognition_session(self, session: FrameProcessor | None) -> None:
        self._recognition = session

    def set_registration_session(self, session: RegistrationProcessor | None) -> None:
        self._registration = session

    @property
    def fps(self) -> float:
        return self._rate.rate()

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    @property
    def error(self) -> str | None:
        return self._error

    def run(self) -> None:
        while not self._stop_event.is_set():
            version = self._source.version
            value = self._source.wait_newer(version, timeout=0.1)
            if value is None:
                continue
            frame = value if isinstance(value, Frame) else None
            if frame is None:
                continue

            if frame.sequence > self._last_sequence + 1:
                self._dropped += frame.sequence - self._last_sequence - 1
            self._last_sequence = frame.sequence

            mode = self.mode
            started = time.perf_counter()
            try:
                output = self._dispatch(frame, mode, started)
            except Exception as exc:
                logger.exception("Inference failed")
                self._error = str(exc)
                output = PipelineOutput(frame=frame, mode=mode)
            self._rate.tick()
            self._sink.set(output)

    def _dispatch(self, frame: Frame, mode: Mode, started: float) -> PipelineOutput:
        if mode is Mode.RECOGNIZE and self._recognition is not None:
            result = self._recognition.process(frame.image)
            return PipelineOutput(
                frame=frame,
                mode=mode,
                recognition=result,
                inference_ms=(time.perf_counter() - started) * 1000.0,
            )

        if mode is Mode.REGISTER and self._registration is not None:
            attempt = self._registration.submit_frame(frame.image)
            return PipelineOutput(
                frame=frame,
                mode=mode,
                attempt=attempt,
                inference_ms=(time.perf_counter() - started) * 1000.0,
            )

        return PipelineOutput(
            frame=frame,
            mode=mode,
            inference_ms=(time.perf_counter() - started) * 1000.0,
        )


@dataclass(frozen=True)
class PipelineStats:
    """Live counters for the status bar and the benchmark."""

    mode: Mode
    capture_fps: float
    inference_fps: float
    display_fps: float
    dropped_frames: int
    error: str | None = None


class RecognitionPipeline:
    """Owns the capture and inference threads and the current mode."""

    def __init__(self, config: AppConfig, source: CaptureSource) -> None:
        self._config = config
        self._source = source
        self._frames: LatestValue = LatestValue()
        self._outputs: LatestValue = LatestValue()
        self._capture = CaptureThread(source, self._frames, config)
        self._inference = InferenceThread(self._frames, self._outputs, config)
        self._display_rate = RateCounter()
        self._output_version = 0
        self._closed = False

    @property
    def mode(self) -> Mode:
        return self._inference.mode

    def describe_source(self) -> str:
        return self._source.describe()

    def start(self) -> None:
        self._capture.start()
        self._inference.start()

    def set_mode(self, mode: Mode) -> None:
        self._inference.set_mode(mode)

    def set_recognition_session(self, session: FrameProcessor | None) -> None:
        self._inference.set_recognition_session(session)

    def begin_registration(self, session: RegistrationProcessor) -> None:
        self._inference.set_registration_session(session)
        self._inference.set_mode(Mode.REGISTER)

    def latest_frame(self) -> Frame | None:
        value = self._frames.peek()
        return value if isinstance(value, Frame) else None

    def wait_output(self, timeout: float = 0.1) -> PipelineOutput | None:
        """Wait for the next processed output."""
        value = self._outputs.wait_newer(self._output_version, timeout)
        if value is None:
            return None
        self._output_version = self._outputs.version
        return value if isinstance(value, PipelineOutput) else None

    def note_displayed(self) -> None:
        """Count one rendered frame, so display FPS reflects real painting."""
        self._display_rate.tick()

    def stats(self) -> PipelineStats:
        return PipelineStats(
            mode=self.mode,
            capture_fps=self._capture.fps,
            inference_fps=self._inference.fps,
            display_fps=self._display_rate.rate(),
            dropped_frames=self._inference.dropped_frames,
            error=self._inference.error or self._capture.error,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._inference.stop()
        self._capture.stop()
        self._frames.close()
        self._outputs.close()
        for thread in (self._inference, self._capture):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._source.close()

    def __enter__(self) -> RecognitionPipeline:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class DisplayPump:
    """Adapts `wait_output` to a polled UI loop.

    Qt timers cannot block, so the GUI asks this for the newest output without
    waiting; the CLI waits. Keeping the branch here means the GUI never holds a
    lock while painting.
    """

    def __init__(self, pipeline: RecognitionPipeline) -> None:
        self._pipeline = pipeline

    def poll(self) -> PipelineOutput | None:
        return self._pipeline.wait_output(timeout=0.0)

    def wait(self, timeout: float = 0.1) -> PipelineOutput | None:
        return self._pipeline.wait_output(timeout=timeout)

    def frames(
        self, stop: Callable[[], bool], timeout: float = 0.1
    ) -> list[PipelineOutput]:
        """Collect outputs until ``stop()`` returns True (CLI helper)."""
        collected: list[PipelineOutput] = []
        while not stop():
            output = self.wait(timeout)
            if output is not None:
                collected.append(output)
        return collected
