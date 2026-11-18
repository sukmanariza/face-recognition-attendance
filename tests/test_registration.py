"""Staged enrollment: what gets accepted, rejected, and stored."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import RegistrationConfig
from src.database import FaceDatabase
from src.enrollment_plan import PoseKind, build_plan
from src.face_registration import (
    FaceRegistrar,
    FaceRegistrationSession,
    RegistrationError,
    SampleStatus,
)
from src.presentation import SpoofLevel
from src.similarity import GalleryIndex, IdentityRecord
from tests.support import (
    StubLiveness,
    make_face,
    make_landmarks,
    make_report,
    unit_vector,
)

POSES: dict[PoseKind, tuple[float, float]] = {
    PoseKind.FRONTAL: (0.0, 0.0),
    PoseKind.TURN_LEFT: (-0.3, 0.0),
    PoseKind.TURN_RIGHT: (0.3, 0.0),
    PoseKind.TILT: (0.0, 12.0),
    PoseKind.CLOSER: (0.0, 0.0),
}


class ScriptedDetector:
    """Detector double that returns one pose per frame and fixed embeddings."""

    def __init__(self, poses: list[PoseKind], faces_per_frame: int = 1) -> None:
        self.model_signature = "stub:test"
        self._poses = poses
        self._faces_per_frame = faces_per_frame
        self._index = 0

    @property
    def is_loaded(self) -> bool:
        return True

    def detect(self, frame, max_faces: int = 0):
        if self._index >= len(self._poses):
            return []
        yaw, roll = POSES[self._poses[self._index]]
        return [
            make_face(landmarks=make_landmarks(yaw, roll))
            for _ in range(self._faces_per_frame)
        ]

    def embed(self, frame, faces) -> None:
        yaw, roll = POSES[self._poses[min(self._index, len(self._poses) - 1)]]
        self._index += 1
        base = unit_vector(11)
        for face in faces:
            noise = unit_vector(int(abs(yaw * 100) + abs(roll)) or 12) * 0.02
            face.embedding = base + noise


class ScriptedAssessor:
    def __init__(self, reports: list) -> None:
        self._reports = reports

    def assess(self, frame, face):
        return self._reports.pop(0)


def all_pose_reports(size: int = 220) -> list:
    """One report per stage, matching each pose and growing for the closer stage."""
    sizes = {
        PoseKind.FRONTAL: size,
        PoseKind.TURN_LEFT: size,
        PoseKind.TURN_RIGHT: size,
        PoseKind.TILT: size,
        PoseKind.CLOSER: int(size * 1.4),
    }
    return [
        make_report(
            yaw_ratio=POSES[kind][0],
            roll_deg=POSES[kind][1],
            face_size_px=sizes[kind],
        )
        for kind in (
            PoseKind.FRONTAL,
            PoseKind.TURN_LEFT,
            PoseKind.TURN_RIGHT,
            PoseKind.TILT,
            PoseKind.CLOSER,
        )
    ]


def make_session(
    detector, reports, config: RegistrationConfig | None = None, gallery=None
) -> FaceRegistrationSession:
    return FaceRegistrationSession(
        detector,  # type: ignore[arg-type]
        ScriptedAssessor(reports),  # type: ignore[arg-type]
        config or RegistrationConfig(capture_interval_seconds=0.0),
        None,
        gallery,
    )


def frame() -> np.ndarray:
    return np.full((240, 320, 3), 120, dtype=np.uint8)


class StagedCaptureTests(unittest.TestCase):
    def test_five_matching_poses_complete_the_capture(self) -> None:
        poses = [stage.kind for stage in build_plan(RegistrationConfig())]
        session = make_session(ScriptedDetector(poses), all_pose_reports())
        statuses = [session.submit_frame(frame()).status for _ in range(5)]
        self.assertEqual(statuses, [SampleStatus.CAPTURED] * 4 + [SampleStatus.COMPLETE])
        self.assertTrue(session.is_complete)
        self.assertEqual(session.collected, 5)

    def test_a_wrong_pose_does_not_capture(self) -> None:
        detector = ScriptedDetector([PoseKind.FRONTAL])
        reports = [make_report(yaw_ratio=0.4)]
        session = make_session(detector, reports)
        attempt = session.submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.REJECTED)
        self.assertEqual(session.collected, 0)

    def test_stage_advances_only_after_an_accepted_sample(self) -> None:
        detector = ScriptedDetector([PoseKind.FRONTAL, PoseKind.FRONTAL])
        reports = [make_report(yaw_ratio=0.0), make_report(yaw_ratio=0.0)]
        session = make_session(detector, reports)
        session.submit_frame(frame())
        self.assertEqual(session.current_stage.kind, PoseKind.TURN_LEFT)  # type: ignore[union-attr]

    def test_failed_quality_does_not_capture(self) -> None:
        detector = ScriptedDetector([PoseKind.FRONTAL])
        reports = [make_report(passed=False, reasons=("blurry",))]
        session = make_session(detector, reports)
        attempt = session.submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.REJECTED)
        self.assertIn("blurry", attempt.message)

    def test_no_face_is_reported(self) -> None:
        session = make_session(ScriptedDetector([]), [])
        attempt = session.submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.REJECTED)
        self.assertIn("no face", attempt.message.lower())

    def test_two_faces_are_rejected(self) -> None:
        session = make_session(ScriptedDetector([PoseKind.FRONTAL], faces_per_frame=2), [])
        attempt = session.submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.REJECTED)
        self.assertIn("2 faces", attempt.message)

    def test_capture_interval_makes_the_session_wait(self) -> None:
        session = make_session(
            ScriptedDetector([PoseKind.FRONTAL, PoseKind.TURN_LEFT]),
            all_pose_reports(),
            RegistrationConfig(capture_interval_seconds=5.0),
        )
        first = session.submit_frame(frame())
        second = session.submit_frame(frame())
        self.assertEqual(first.status, SampleStatus.CAPTURED)
        self.assertEqual(second.status, SampleStatus.WAITING)

    def test_progress_is_reported(self) -> None:
        poses = [stage.kind for stage in build_plan(RegistrationConfig())]
        session = make_session(ScriptedDetector(poses), all_pose_reports())
        attempt = session.submit_frame(frame())
        self.assertAlmostEqual(attempt.progress, 0.2)
        self.assertEqual(attempt.required, 5)

    def test_frozen_session_stops_capturing(self) -> None:
        session = make_session(ScriptedDetector([PoseKind.FRONTAL]), all_pose_reports())
        session.freeze()
        self.assertEqual(session.submit_frame(frame()).status, SampleStatus.COMPLETE)
        self.assertEqual(session.collected, 0)

    def test_reset_clears_everything(self) -> None:
        detector = ScriptedDetector([PoseKind.FRONTAL])
        session = make_session(detector, [make_report(yaw_ratio=0.0)])
        session.submit_frame(frame())
        session.reset()
        self.assertEqual(session.collected, 0)
        self.assertFalse(session.is_frozen)

    def test_inconsistent_sample_is_rejected(self) -> None:
        detector = ScriptedDetector([PoseKind.FRONTAL, PoseKind.TURN_LEFT])
        reports = all_pose_reports()[:2]
        session = make_session(detector, reports)
        session.submit_frame(frame())

        # Replace the next embedding with an unrelated vector.
        original_embed = detector.embed

        def unrelated(frame_arg, faces):
            original_embed(frame_arg, faces)
            for face in faces:
                face.embedding = unit_vector(99)

        detector.embed = unrelated  # type: ignore[method-assign]
        attempt = session.submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.REJECTED)
        self.assertIn("same person", attempt.message)


class BuildSamplesTests(unittest.TestCase):
    def test_incomplete_capture_cannot_be_built(self) -> None:
        session = make_session(ScriptedDetector([PoseKind.FRONTAL]), all_pose_reports())
        session.submit_frame(frame())
        with self.assertRaises(RegistrationError):
            session.build_samples()

    def test_complete_capture_returns_every_sample(self) -> None:
        poses = [stage.kind for stage in build_plan(RegistrationConfig())]
        session = make_session(ScriptedDetector(poses), all_pose_reports())
        for _ in range(5):
            session.submit_frame(frame())
        samples = session.build_samples()
        self.assertEqual(len(samples), 5)
        self.assertTrue(all(sample.model_signature == "stub:test" for sample in samples))

    def test_min_consistency_is_high_for_similar_samples(self) -> None:
        poses = [stage.kind for stage in build_plan(RegistrationConfig())]
        session = make_session(ScriptedDetector(poses), all_pose_reports())
        for _ in range(5):
            session.submit_frame(frame())
        self.assertGreater(session.min_consistency(), 0.9)

    def test_duplicate_of_an_existing_identity_is_rejected(self) -> None:
        records = [IdentityRecord("001", "Alice", unit_vector(11).reshape(1, -1))]
        gallery = GalleryIndex(records)
        session = make_session(
            ScriptedDetector([PoseKind.FRONTAL]), all_pose_reports(), gallery=gallery
        )
        attempt = session.submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.REJECTED)
        self.assertIn("already registered", attempt.message)


class RegistrarTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.database = FaceDatabase(Path(self._directory.name) / "faces.db")
        self.addCleanup(self.database.close)
        self.config = RegistrationConfig(capture_interval_seconds=0.0)
        self.registrar = FaceRegistrar(self.database, self.config)

    def _capture(self) -> FaceRegistrationSession:
        poses = [stage.kind for stage in build_plan(self.config)]
        session = make_session(ScriptedDetector(poses), all_pose_reports(), self.config)
        for _ in range(len(poses)):
            session.submit_frame(frame())
        return session

    def test_saving_a_new_identity(self) -> None:
        session = self._capture()
        result = self.registrar.save_new("001", "Ana", session.build_samples(), session.min_consistency())
        self.assertEqual(result.employee_id, "001")
        self.assertEqual(result.sample_count, 5)
        self.assertFalse(result.replaced_existing)
        self.assertIn("registered", result.describe())

    def test_saving_the_same_face_twice_is_refused(self) -> None:
        session = self._capture()
        self.registrar.save_new("001", "Ana", session.build_samples(), 1.0)
        second = self._capture()
        with self.assertRaises(RegistrationError):
            self.registrar.save_new("002", "Other", second.build_samples(), 1.0)

    def test_updating_an_identity_replaces_its_samples(self) -> None:
        session = self._capture()
        self.registrar.save_new("001", "Ana", session.build_samples(), 1.0)
        update = self._capture()
        result = self.registrar.save_update("001", update.build_samples(), 1.0)
        self.assertTrue(result.replaced_existing)
        self.assertEqual(self.database.sample_count("001"), 5)

    def test_updating_can_rename_at_the_same_time(self) -> None:
        session = self._capture()
        self.registrar.save_new("001", "Ana", session.build_samples(), 1.0)
        update = self._capture()
        result = self.registrar.save_update("001", update.build_samples(), 1.0, full_name="Ana Maria")
        self.assertEqual(result.full_name, "Ana Maria")

class PresentationGuardTests(unittest.TestCase):
    def _session(self, level: SpoofLevel) -> FaceRegistrationSession:
        return FaceRegistrationSession(
            ScriptedDetector([PoseKind.FRONTAL]),  # type: ignore[arg-type]
            ScriptedAssessor([make_report(yaw_ratio=0.0)]),  # type: ignore[arg-type]
            RegistrationConfig(capture_interval_seconds=0.0),
            StubLiveness(level),  # type: ignore[arg-type]
        )

    def test_a_photo_is_refused_during_enrollment(self) -> None:
        attempt = self._session(SpoofLevel.SUSPECTED).submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.REJECTED)
        self.assertIn("photo or a screen", attempt.message)
        self.assertIs(attempt.presentation.level, SpoofLevel.SUSPECTED)  # type: ignore[union-attr]

    def test_a_real_face_is_captured(self) -> None:
        attempt = self._session(SpoofLevel.GENUINE).submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.CAPTURED)
        self.assertIs(attempt.presentation.level, SpoofLevel.GENUINE)  # type: ignore[union-attr]

    def test_an_undecided_verdict_does_not_block(self) -> None:
        attempt = self._session(SpoofLevel.UNCERTAIN).submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.CAPTURED)

    def test_the_verdict_is_reported_even_when_quality_fails(self) -> None:
        session = FaceRegistrationSession(
            ScriptedDetector([PoseKind.FRONTAL]),  # type: ignore[arg-type]
            ScriptedAssessor([make_report(passed=False, reasons=("blurry",))]),  # type: ignore[arg-type]
            RegistrationConfig(capture_interval_seconds=0.0),
            StubLiveness(SpoofLevel.GENUINE),  # type: ignore[arg-type]
        )
        attempt = session.submit_frame(frame())
        self.assertEqual(attempt.status, SampleStatus.REJECTED)
        self.assertIsNotNone(attempt.presentation)


if __name__ == "__main__":
    unittest.main()
