"""Pipeline plumbing: latest-frame handoff, threads, and shutdown."""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from config import load_config
from src.camera import Frame
from src.face_detector import DetectedFace
from src.face_recognizer import FaceState, FrameRecognition, RecognizedFace
from src.face_registration import SampleAttempt, SampleStatus
from src.pipeline import (
    CaptureThread,
    DisplayPump,
    ImageSource,
    LatestValue,
    Mode,
    PipelineOutput,
    RecognitionPipeline,
)
from tests.support import make_face


class LatestValueTests(unittest.TestCase):
    def test_newest_value_wins(self) -> None:
        slot = LatestValue()
        slot.set(1)
        slot.set(2)
        self.assertEqual(slot.get(), 2)

    def test_version_increases(self) -> None:
        slot = LatestValue()
        slot.set("a")
        first = slot.version
        slot.set("b")
        self.assertGreater(slot.version, first)

    def test_wait_returns_a_newer_value(self) -> None:
        slot = LatestValue()
        slot.set("old")
        slot.set("new")
        self.assertEqual(slot.wait_newer(1, timeout=0.1), "new")

    def test_wait_times_out_without_new_data(self) -> None:
        slot = LatestValue()
        slot.set("only")
        self.assertIsNone(slot.wait_newer(5, timeout=0.01))

    def test_close_releases_waiters(self) -> None:
        slot = LatestValue()
        slot.close()
        self.assertIsNone(slot.wait_newer(0, timeout=0.01))


class ImageSourceTests(unittest.TestCase):
    def test_repeating_source_never_ends(self) -> None:
        source = ImageSource(np.zeros((120, 160, 3), dtype=np.uint8))
        self.assertIsNotNone(source.read())
        self.assertIsNotNone(source.read())

    def test_non_repeating_source_ends(self) -> None:
        source = ImageSource(np.zeros((120, 160, 3), dtype=np.uint8), repeat=False)
        self.assertIsNotNone(source.read())
        self.assertIsNone(source.read())

    def test_empty_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ImageSource(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_describe_mentions_the_size(self) -> None:
        source = ImageSource(np.zeros((120, 160, 3), dtype=np.uint8), label="still")
        self.assertIn("160x120", source.describe())

    def test_close_stops_the_source(self) -> None:
        source = ImageSource(np.zeros((8, 8, 3), dtype=np.uint8))
        source.close()
        self.assertIsNone(source.read())


class CaptureThreadTests(unittest.TestCase):
    def test_frames_reach_the_sink(self) -> None:
        config = load_config()
        sink = LatestValue()
        source = ImageSource(np.full((60, 80, 3), 90, dtype=np.uint8))
        thread = CaptureThread(source, sink, config)
        thread.start()
        self.addCleanup(thread.stop)
        deadline = time.monotonic() + 2.0
        while sink.get() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        thread.stop()
        thread.join(timeout=1.0)
        value = sink.get()
        self.assertIsInstance(value, Frame)

    def test_stopping_ends_the_thread(self) -> None:
        config = load_config()
        thread = CaptureThread(
            ImageSource(np.zeros((60, 80, 3), dtype=np.uint8)), LatestValue(), config
        )
        thread.start()
        thread.stop()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())


class _NullProcessor:
    def process(self, image, timestamp=None):
        recognized = RecognizedFace(
            face=make_face(),
            track_id=1,
            state=FaceState.MATCHED,
            similarity=0.7,
            similarity_smoothed=0.7,
            hits=3,
            employee_id="001",
            full_name="Alice",
        )
        return FrameRecognition(faces=(recognized,), inference_ms=1.0, fps=20.0, threshold=0.45)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.source = ImageSource(np.full((90, 120, 3), 100, dtype=np.uint8))
        self.pipeline = RecognitionPipeline(self.config, self.source)
        self.addCleanup(self.pipeline.close)

    def test_output_is_produced_in_idle_mode(self) -> None:
        self.pipeline.start()
        output = self.pipeline.wait_output(timeout=3.0)
        self.assertIsNotNone(output)
        assert output is not None
        self.assertEqual(output.mode, Mode.IDLE)
        self.assertIsNone(output.recognition)

    def test_recognition_mode_runs_the_processor(self) -> None:
        self.pipeline.set_recognition_session(_NullProcessor())  # type: ignore[arg-type]
        self.pipeline.set_mode(Mode.RECOGNIZE)
        self.pipeline.start()
        deadline = time.monotonic() + 3.0
        found = None
        while time.monotonic() < deadline and found is None:
            output = self.pipeline.wait_output(timeout=0.5)
            if output is not None and output.recognition is not None:
                found = output
        self.assertIsNotNone(found, "no recognition output was produced")
        assert found is not None and found.recognition is not None
        self.assertEqual(found.recognition.face_count, 1)

    def test_stats_report_live_counters(self) -> None:
        self.pipeline.start()
        self.pipeline.wait_output(timeout=3.0)
        stats = self.pipeline.stats()
        self.assertIn(stats.mode, tuple(Mode))
        self.assertGreaterEqual(stats.capture_fps, 0.0)

    def test_note_displayed_counts_as_display(self) -> None:
        self.pipeline.note_displayed()
        self.assertGreaterEqual(self.pipeline.stats().display_fps, 0.0)

    def test_describe_source_delegates(self) -> None:
        self.assertIn("120x90", self.pipeline.describe_source())

    def test_latest_frame_is_available(self) -> None:
        self.pipeline.start()
        self.pipeline.wait_output(timeout=3.0)
        frame = self.pipeline.latest_frame()
        self.assertIsNotNone(frame)

    def test_close_is_idempotent(self) -> None:
        self.pipeline.start()
        self.pipeline.close()
        self.pipeline.close()


class DisplayPumpTests(unittest.TestCase):
    def test_poll_never_blocks_without_data(self) -> None:
        pipeline = RecognitionPipeline(
            load_config(), ImageSource(np.zeros((40, 60, 3), dtype=np.uint8))
        )
        self.addCleanup(pipeline.close)
        pump = DisplayPump(pipeline)
        self.assertIsNone(pump.poll())


class SkippedSequenceTests(unittest.TestCase):
    """The camera numbers skipped frames, and that must not park the inference thread.

    ``Camera.read`` returns the frame after the newest one it saw, so sequences jump.
    Waiting for a newer value has to use the handoff slot's own counter: comparing it
    with a frame number stalls the loop and turns a 14 fps preview into 2 fps.
    """

    class _Session:
        def __init__(self) -> None:
            self.count = 0
            self.done = threading.Event()

        def submit_frame(self, frame, /):
            self.count += 1
            if self.count >= 20:
                self.done.set()
            return SampleAttempt(
                status=SampleStatus.WAITING,
                message="hold still",
                collected=0,
                required=5,
                prompt="Front",
            )

    def setUp(self) -> None:
        from src.pipeline import InferenceThread

        self.frames = LatestValue()
        self.outputs = LatestValue()
        self.thread = InferenceThread(self.frames, self.outputs, load_config())
        self.session = self._Session()
        self.thread.set_registration_session(self.session)  # type: ignore[arg-type]
        self.thread.set_mode(Mode.REGISTER)
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)

    def _start_thread(self) -> None:
        self.thread.start()

    def tearDown(self) -> None:
        self.thread.stop()
        self.thread.join(timeout=1.0)
        self.frames.close()
        self.outputs.close()

    def test_a_gap_in_frame_numbers_does_not_stall_processing(self) -> None:
        self._start_thread()
        for index in range(1, 21):
            self.frames.set(Frame(image=self.image, sequence=index * 5))
            time.sleep(0.01)
            if self.session.done.is_set():
                break
        self.assertTrue(
            self.session.done.wait(5.0),
            f"only {self.session.count} of 20 frames were processed",
        )

    def test_skipped_frame_numbers_are_counted_as_dropped(self) -> None:
        self._start_thread()
        for sequence in (1, 4, 7, 10):
            before = self.session.count
            self.frames.set(Frame(image=self.image, sequence=sequence))
            deadline = time.monotonic() + 3.0
            while self.session.count == before and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(self.session.count, before + 1, "frame was not picked up in time")
        # Two frames are missing between each pair: 4-1-1, then 7-4-1 and 10-7-1.
        self.assertEqual(self.thread.dropped_frames, 6)



class RegistrationDispatchTests(unittest.TestCase):
    def test_registration_mode_runs_the_session(self) -> None:
        class _Session:
            def __init__(self) -> None:
                self.calls = 0

            def submit_frame(self, frame, /):
                self.calls += 1
                return SampleAttempt(
                    status=SampleStatus.WAITING,
                    message="hold still",
                    collected=0,
                    required=5,
                    prompt="Front",
                )

            def reset(self) -> None:
                self.calls = 0

        session = _Session()
        pipeline = RecognitionPipeline(
            load_config(), ImageSource(np.full((60, 80, 3), 90, dtype=np.uint8))
        )
        self.addCleanup(pipeline.close)
        pipeline.begin_registration(session)  # type: ignore[arg-type]
        pipeline.start()
        deadline = time.monotonic() + 3.0
        attempt = None
        while time.monotonic() < deadline and attempt is None:
            output = pipeline.wait_output(timeout=0.5)
            if output is not None:
                attempt = output.attempt
        self.assertIsNotNone(attempt)
        self.assertGreater(session.calls, 0)


class OutputDataclassTests(unittest.TestCase):
    def test_defaults(self) -> None:
        output = PipelineOutput(frame=Frame(np.zeros((4, 4, 3), dtype=np.uint8), 1), mode=Mode.IDLE)
        self.assertIsNone(output.recognition)
        self.assertIsNone(output.attempt)
        self.assertEqual(output.inference_ms, 0.0)

    def test_detected_face_is_usable_in_a_recognition_result(self) -> None:
        face = DetectedFace(
            bbox=np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32), det_score=0.9, landmarks=None
        )
        self.assertEqual(face.area, 100.0)


if __name__ == "__main__":
    unittest.main()
