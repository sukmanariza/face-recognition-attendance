"""Recognition session: confirmation, tracking, throttling, and percentages."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from config import AttendanceConfig, RecognitionConfig
from src.attendance import AttendanceService
from src.database import FaceDatabase, FaceSample
from src.face_recognizer import FaceRecognitionSession, FaceState
from src.presentation import SpoofLevel
from src.similarity import GalleryIndex, IdentityRecord, to_percentage
from tests.support import StubLiveness, make_face, unit_vector

ALICE = unit_vector(11)


def gallery() -> GalleryIndex:
    return GalleryIndex([IdentityRecord("001", "Alice", ALICE.reshape(1, -1))])


class FixedDetector:
    def __init__(self, embedding: np.ndarray, faces: int = 1) -> None:
        self.model_signature = "stub:test"
        self._embedding = embedding
        self._faces = faces
        self.detect_calls = 0
        self.embed_calls = 0

    @property
    def is_loaded(self) -> bool:
        return True

    def detect(self, frame, max_faces: int = 0):
        self.detect_calls += 1
        return [make_face() for _ in range(self._faces)]

    def embed(self, frame, faces) -> None:
        self.embed_calls += 1
        for face in faces:
            face.embedding = self._embedding


def config(**overrides) -> RecognitionConfig:
    base = {
        "similarity_threshold": 0.3,
        "detection_interval": 1,
        "embedding_interval_seconds": 0.0,
        "confirmation_hits": 3,
        "smoothing_alpha": 0.65,
        "track_max_age_seconds": 1.5,
    }
    base.update(overrides)
    return RecognitionConfig(**base)


def frame() -> np.ndarray:
    return np.full((240, 320, 3), 120, dtype=np.uint8)


class MatchingTests(unittest.TestCase):
    def test_name_appears_only_after_confirmation_hits(self) -> None:
        session = FaceRecognitionSession(FixedDetector(ALICE), gallery(), config())  # type: ignore[arg-type]
        first = session.process(frame(), timestamp=1.0)
        self.assertEqual(first.faces[0].state, FaceState.VERIFYING)
        self.assertEqual(first.faces[0].display_name, "Verifying")

        for index in range(2):
            result = session.process(frame(), timestamp=1.1 + index * 0.1)
        self.assertEqual(result.faces[0].state, FaceState.MATCHED)
        self.assertEqual(result.faces[0].display_name, "Alice")
        self.assertEqual(result.faces[0].display_id, "001")

    def test_unknown_face_is_not_named(self) -> None:
        # The opposite vector is guaranteed to score -1, which no threshold accepts.
        detector = FixedDetector(-ALICE)
        session = FaceRecognitionSession(detector, gallery(), config())  # type: ignore[arg-type]
        result = session.process(frame(), timestamp=1.0)
        self.assertEqual(result.faces[0].state, FaceState.UNKNOWN)
        self.assertEqual(result.faces[0].display_name, "Unknown")
        self.assertEqual(result.matched, ())

    def test_percentage_matches_the_similarity(self) -> None:
        session = FaceRecognitionSession(FixedDetector(ALICE), gallery(), config())  # type: ignore[arg-type]
        face = session.process(frame(), timestamp=1.0).faces[0]
        self.assertEqual(face.percentage, to_percentage(face.similarity_smoothed))
        self.assertGreaterEqual(face.percentage, 95)

    def test_label_joins_name_and_percentage(self) -> None:
        session = FaceRecognitionSession(FixedDetector(ALICE), gallery(), config())  # type: ignore[arg-type]
        face = session.process(frame(), timestamp=1.0).faces[0]
        self.assertIn("%", face.label)
        self.assertIn(face.display_name, face.label)

    def test_empty_gallery_yields_unknown(self) -> None:
        session = FaceRecognitionSession(FixedDetector(ALICE), GalleryIndex([]), config())  # type: ignore[arg-type]
        face = session.process(frame(), timestamp=1.0).faces[0]
        self.assertFalse(face.accepted)

    def test_hits_reset_when_the_match_is_lost(self) -> None:
        detector = FixedDetector(ALICE)
        session = FaceRecognitionSession(detector, gallery(), config())  # type: ignore[arg-type]
        session.process(frame(), timestamp=1.0)
        detector._embedding = -ALICE
        result = session.process(frame(), timestamp=1.1)
        self.assertEqual(result.faces[0].hits, 0)


class ThrottleTests(unittest.TestCase):
    def test_detection_runs_once_per_interval(self) -> None:
        detector = FixedDetector(ALICE, faces=1)
        session = FaceRecognitionSession(
            detector, gallery(), config(detection_interval=3)  # type: ignore[arg-type]
        )
        for index in range(6):
            session.process(frame(), timestamp=1.0 + index * 0.1)
        # Frame 1 has no cached boxes, then frames 3 and 6 detect.
        self.assertEqual(detector.detect_calls, 3)

    def test_embedding_respects_the_interval(self) -> None:
        detector = FixedDetector(ALICE)
        session = FaceRecognitionSession(
            detector, gallery(), config(embedding_interval_seconds=10.0)  # type: ignore[arg-type]
        )
        session.process(frame(), timestamp=1.0)
        session.process(frame(), timestamp=1.1)
        self.assertEqual(detector.embed_calls, 1)

    def test_tracks_are_pruned_after_the_max_age(self) -> None:
        detector = FixedDetector(ALICE)
        session = FaceRecognitionSession(detector, gallery(), config())  # type: ignore[arg-type]
        session.process(frame(), timestamp=1.0)
        self.assertEqual(session.track_count, 1)
        detector._faces = 0
        session.process(frame(), timestamp=5.0)
        self.assertEqual(session.track_count, 0)


class StateTests(unittest.TestCase):
    def test_frame_reports_face_count_and_hud_fields(self) -> None:
        detector = FixedDetector(ALICE, faces=2)
        session = FaceRecognitionSession(detector, gallery(), config())  # type: ignore[arg-type]
        result = session.process(frame(), timestamp=1.0)
        self.assertEqual(result.face_count, 2)
        self.assertIn("2 face(s)", result.summary())
        self.assertGreaterEqual(result.inference_ms, 0.0)

    def test_threshold_is_exposed(self) -> None:
        session = FaceRecognitionSession(FixedDetector(ALICE), gallery(), config())  # type: ignore[arg-type]
        self.assertAlmostEqual(session.threshold, 0.3)

    def test_update_config_changes_the_threshold(self) -> None:
        session = FaceRecognitionSession(FixedDetector(ALICE), gallery(), config())  # type: ignore[arg-type]
        session.update_config(config(similarity_threshold=0.99))
        self.assertAlmostEqual(session.threshold, 0.99)

    def test_set_gallery_resets_tracking(self) -> None:
        session = FaceRecognitionSession(FixedDetector(ALICE), gallery(), config())  # type: ignore[arg-type]
        session.process(frame(), timestamp=1.0)
        session.set_gallery(GalleryIndex([]))
        self.assertEqual(session.track_count, 0)

    def test_reset_clears_tracks(self) -> None:
        session = FaceRecognitionSession(FixedDetector(ALICE), gallery(), config())  # type: ignore[arg-type]
        session.process(frame(), timestamp=1.0)
        session.reset()
        self.assertEqual(session.track_count, 0)

    def test_no_faces_returns_an_empty_result(self) -> None:
        detector = FixedDetector(ALICE, faces=0)
        session = FaceRecognitionSession(detector, gallery(), config())  # type: ignore[arg-type]
        result = session.process(frame(), timestamp=1.0)
        self.assertEqual(result.faces, ())
        self.assertEqual(result.face_count, 0)


class PresentationTests(unittest.TestCase):
    def test_a_suspected_photo_is_reported_but_still_named_in_advisory_mode(self) -> None:
        session = FaceRecognitionSession(
            FixedDetector(ALICE),  # type: ignore[arg-type]
            gallery(),
            config(),
            StubLiveness(SpoofLevel.SUSPECTED),  # type: ignore[arg-type]
        )
        for index in range(3):
            result = session.process(frame(), timestamp=1.0 + index * 0.1)
        face = result.faces[0]
        self.assertEqual(face.state, FaceState.MATCHED)
        presentation = face.presentation
        self.assertIsNotNone(presentation)
        assert presentation is not None
        self.assertTrue(presentation.is_suspected)
        self.assertEqual(presentation.label, "Photo or screen (2D)")

    def test_enforce_mode_refuses_a_detected_photo(self) -> None:
        session = FaceRecognitionSession(
            FixedDetector(ALICE),  # type: ignore[arg-type]
            gallery(),
            config(),
            StubLiveness(SpoofLevel.SUSPECTED),  # type: ignore[arg-type]
            enforce_presentation=True,
        )
        for index in range(3):
            result = session.process(frame(), timestamp=1.0 + index * 0.1)
        face = result.faces[0]
        self.assertEqual(face.state, FaceState.UNKNOWN)
        self.assertEqual(face.display_name, "Unknown")
        self.assertIn("Photo or screen", face.reason)

    def test_a_real_face_is_labelled(self) -> None:
        session = FaceRecognitionSession(
            FixedDetector(ALICE),  # type: ignore[arg-type]
            gallery(),
            config(),
            StubLiveness(SpoofLevel.GENUINE),  # type: ignore[arg-type]
        )
        for index in range(3):
            result = session.process(frame(), timestamp=1.0 + index * 0.1)
        face = result.faces[0]
        self.assertEqual(face.state, FaceState.MATCHED)
        presentation = face.presentation
        self.assertIsNotNone(presentation)
        assert presentation is not None
        self.assertTrue(presentation.is_genuine)
        self.assertIn("[3D]", face.label)

    def test_the_check_runs_even_when_the_embedding_is_throttled(self) -> None:
        liveness = StubLiveness(SpoofLevel.GENUINE)
        session = FaceRecognitionSession(
            FixedDetector(ALICE),  # type: ignore[arg-type]
            gallery(),
            config(embedding_interval_seconds=10.0),
            liveness,  # type: ignore[arg-type]
        )
        session.process(frame(), timestamp=1.0)
        session.process(frame(), timestamp=1.1)
        session.process(frame(), timestamp=1.2)
        self.assertEqual(liveness.calls, 3)

    def test_two_faces_are_not_checked(self) -> None:
        liveness = StubLiveness(SpoofLevel.SUSPECTED)
        session = FaceRecognitionSession(
            FixedDetector(ALICE, faces=2),  # type: ignore[arg-type]
            gallery(),
            config(),
            liveness,  # type: ignore[arg-type]
        )
        result = session.process(frame(), timestamp=1.0)
        self.assertEqual(liveness.calls, 0)
        self.assertTrue(all(face.presentation is None for face in result.faces))

    def test_reset_resets_the_detector(self) -> None:
        liveness = StubLiveness(SpoofLevel.GENUINE)
        session = FaceRecognitionSession(
            FixedDetector(ALICE),  # type: ignore[arg-type]
            gallery(),
            config(),
            liveness,  # type: ignore[arg-type]
        )
        session.reset()
        self.assertEqual(liveness.resets, 1)


class AttendanceIntegrationTests(unittest.TestCase):
    """A confirmed match must reach the attendance table exactly once."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        root = Path(self._directory.name)
        self.database = FaceDatabase(root / "faces.db")
        self.addCleanup(self.database.close)
        self.database.add_employee(
            "001", "Alice", [FaceSample(embedding=ALICE)]
        )
        self.morning = datetime(2026, 11, 16, 7, 50).astimezone()
        self.service = AttendanceService(
            self.database, AttendanceConfig(), lambda: self.morning
        )

    def _session(self) -> FaceRecognitionSession:
        return FaceRecognitionSession(
            FixedDetector(ALICE),  # type: ignore[arg-type]
            gallery(),
            config(),
            None,
            attendance=self.service,
        )

    def _run(self, session: FaceRecognitionSession) -> None:
        for index in range(4):
            session.process(frame(), timestamp=1.0 + index * 0.1)

    def test_a_confirmed_match_is_recorded_once(self) -> None:
        session = self._session()
        self._run(session)
        self._run(session)
        entries = self.database.attendance_for(self.service.today())
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].employee_id, "001")

    def test_the_face_reports_the_attendance_state(self) -> None:
        session = self._session()
        result = None
        for index in range(4):
            result = session.process(frame(), timestamp=1.0 + index * 0.1)
        assert result is not None
        face = result.faces[0]
        self.assertIsNotNone(face.attendance)
        assert face.attendance is not None and face.attendance.mark is not None
        self.assertEqual(face.attendance.mark.entry.status, "present")

    def test_an_unconfirmed_face_records_nothing(self) -> None:
        session = self._session()
        session.process(frame(), timestamp=1.0)
        self.assertEqual(self.database.attendance_for(self.service.today()), [])


if __name__ == "__main__":
    unittest.main()
