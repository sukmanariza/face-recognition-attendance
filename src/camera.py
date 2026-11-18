"""Webcam access with a dedicated capture thread.

Reading frames from ``cv2.VideoCapture`` on the same thread as inference makes
latency pile up: by the time a frame is processed, newer ones are already waiting
in the driver buffer. A capture thread keeps only the newest frame, so inference
always works on fresh data and a slow camera driver cannot freeze the app.

Backend selection matters for portability: DirectShow on Windows, V4L2 on Linux
(including Raspberry Pi and other single-board computers), AVFoundation on macOS.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from config import CameraConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 0.005


def _linux_backend() -> int:
    return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)


def _windows_backend() -> int:
    return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)


def _default_backend() -> int:
    if sys.platform.startswith("win"):
        return _windows_backend()
    if sys.platform.startswith("linux"):
        return _linux_backend()
    return cv2.CAP_ANY


_BACKEND_BY_NAME: dict[str, int] = {
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "v4l2": _linux_backend(),
    "any": cv2.CAP_ANY,
}


# OpenCV 5 moved the log-level API to ``cv2.utils.logging``; other builds keep it
# on ``cv2`` directly. Resolved once at import.
def _resolve_opencv_logging() -> object | None:
    inner = getattr(getattr(cv2, "utils", None), "logging", None)
    if inner is not None and hasattr(inner, "setLogLevel") and hasattr(inner, "getLogLevel"):
        return inner
    if hasattr(cv2, "setLogLevel") and hasattr(cv2, "getLogLevel"):
        return cv2
    return None


_OPENCV_LOGGING = _resolve_opencv_logging()
_LOG_LEVEL_ERROR = int(getattr(_OPENCV_LOGGING, "LOG_LEVEL_ERROR", 2))

# OpenCV spawns its own thread pool for resize and colour conversion. On a 2 to 4
# core board that pool competes with the inference threads and makes both slower, so
# it is pinned to one thread: the operations here work on small images anyway.
with contextlib.suppress(Exception):  # pragma: no cover - older builds expose it too
    cv2.setNumThreads(1)


class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


@dataclass(frozen=True)
class Frame:
    """One camera frame with a monotonically increasing sequence number.

    The sequence is the only freshness marker: consumers compare it to know
    whether something new arrived since their last read.
    """

    image: np.ndarray
    sequence: int


def resolve_backend(name: str) -> int:
    """Translate a backend name from the configuration into an OpenCV constant."""
    if name == "auto":
        return _default_backend()
    try:
        return _BACKEND_BY_NAME[name]
    except KeyError as exc:  # pragma: no cover - prevented by config validation
        raise CameraError(f"Unknown camera backend: {name}") from exc


class Camera:
    """A webcam read continuously on a background thread."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._latest: Frame | None = None
        self._sequence = 0
        self._consecutive_failures = 0
        self._error: CameraError | None = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def describe(self) -> str:
        if not self.is_open:
            return f"Camera {self._config.index}: not active"
        width, height = self.resolution()
        return f"Camera {self._config.index}: {width}x{height} @ {self.measured_fps():.0f} FPS"

    def resolution(self) -> tuple[int, int]:
        if self._capture is None:
            return (0, 0)
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def measured_fps(self) -> float:
        if self._capture is None:
            return 0.0
        return float(self._capture.get(cv2.CAP_PROP_FPS))

    def open(self) -> None:
        if self._thread is not None:
            return
        backend = resolve_backend(self._config.backend)
        capture = cv2.VideoCapture(self._config.index, backend)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Camera index {self._config.index} could not be opened "
                f"(backend: {self._config.backend}). Check that the camera is connected, "
                "not used by another application, and allowed in the system privacy settings."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.requested_fps)
        # A short driver buffer keeps ``read()`` from returning stale frames.
        with contextlib.suppress(AttributeError, cv2.error):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = capture
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()

        for _ in range(self._config.warmup_frames):
            self.read(timeout=self._config.read_timeout_seconds)

        width, height = self.resolution()
        logger.info("Camera %s opened (%dx%d)", self._config.index, width, height)

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        with self._condition:
            self._latest = None
            self._condition.notify_all()

    def read(self, timeout: float | None = None) -> Frame:
        """Wait for a frame newer than the last one returned."""
        timeout = self._config.read_timeout_seconds if timeout is None else timeout
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._condition:
            latest = self._latest
            seen_sequence = latest.sequence if latest is not None else 0
            while True:
                if self._error is not None:
                    raise self._error
                candidate = self._latest
                if candidate is not None and candidate.sequence > seen_sequence:
                    return candidate
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraError(f"No new frame within {timeout:.1f} s.")
                if not self._condition.wait(timeout=remaining) and not self.is_open:
                    raise self._error or CameraError("The capture thread stopped.")

    def _capture_loop(self) -> None:
        capture = self._capture
        if capture is None:  # pragma: no cover - only reachable on misuse
            return
        while not self._stop_event.is_set():
            ok, image = capture.read()
            if not ok or image is None:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._config.max_consecutive_failures:
                    self._error = CameraError(
                        f"Camera stopped delivering frames after {self._consecutive_failures} "
                        "failed reads."
                    )
                    with self._condition:
                        self._condition.notify_all()
                    return
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            self._consecutive_failures = 0
            self._sequence += 1
            frame = Frame(image=image, sequence=self._sequence)
            with self._condition:
                self._latest = frame
                self._condition.notify_all()


def probe_cameras(max_index: int = 4, backend: str = "auto", probe_read: bool = True) -> list[int]:
    """Return the camera indices that can be opened.

    OpenCV prints a warning for indices that do not exist; the log level is
    raised for the duration of the probe so ``status`` output stays readable.
    """
    available: list[int] = []
    resolved = resolve_backend(backend)
    previous_level = _set_opencv_log_level_silent()
    try:
        for index in range(max_index + 1):
            capture = cv2.VideoCapture(index, resolved)
            try:
                if not capture.isOpened():
                    continue
                if probe_read:
                    ok, _ = capture.read()
                    if not ok:
                        continue
                available.append(index)
            finally:
                capture.release()
    finally:
        _restore_opencv_log_level(previous_level)
    return available


def _set_opencv_log_level_silent() -> tuple[object, int] | None:
    if _OPENCV_LOGGING is None:  # pragma: no cover - build without the log API
        return None
    try:
        previous = int(_OPENCV_LOGGING.getLogLevel())  # type: ignore[attr-defined]
        _OPENCV_LOGGING.setLogLevel(_LOG_LEVEL_ERROR)  # type: ignore[attr-defined]
        return (_OPENCV_LOGGING, previous)
    except (AttributeError, TypeError, cv2.error):  # pragma: no cover
        return None


def _restore_opencv_log_level(state: tuple[object, int] | None) -> None:
    if state is None:
        return
    target, level = state
    with contextlib.suppress(AttributeError, TypeError, cv2.error):  # pragma: no cover
        target.setLogLevel(level)  # type: ignore[attr-defined]
