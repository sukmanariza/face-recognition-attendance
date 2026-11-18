"""Quality gate: pose estimation and the accept/reject decision."""

from __future__ import annotations

import unittest

import numpy as np

from config import QualityConfig
from src.quality import FaceQualityAssessor, estimate_pose
from tests.support import make_face, make_landmarks, textured_frame


class PoseTests(unittest.TestCase):
    def test_frontal_face_has_no_yaw_or_roll(self) -> None:
        yaw, roll = estimate_pose(make_landmarks(0.0, 0.0))
        self.assertAlmostEqual(yaw, 0.0, places=3)
        self.assertAlmostEqual(roll, 0.0, places=3)

    def test_roll_matches_the_requested_angle(self) -> None:
        _yaw, roll = estimate_pose(make_landmarks(0.0, 15.0))
        self.assertAlmostEqual(roll, 15.0, places=2)

    def test_yaw_matches_the_requested_ratio(self) -> None:
        yaw, _ = estimate_pose(make_landmarks(0.4, 0.0))
        self.assertAlmostEqual(yaw, 0.4, places=2)

    def test_yaw_sign_is_opposite_for_the_other_side(self) -> None:
        left_yaw, _ = estimate_pose(make_landmarks(0.4, 0.0))
        right_yaw, _ = estimate_pose(make_landmarks(-0.4, 0.0))
        self.assertGreater(left_yaw, 0.0)
        self.assertLess(right_yaw, 0.0)

    def test_malformed_landmarks_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_pose(np.zeros((4, 2), dtype=np.float32))


class AssessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = QualityConfig()
        self.assessor = FaceQualityAssessor(self.config)
        self.frame = textured_frame()

    def test_good_frame_passes(self) -> None:
        report = self.assessor.assess(self.frame, make_face((60.0, 60.0, 260.0, 260.0)))
        self.assertTrue(report.passed, report.reasons)
        self.assertGreaterEqual(report.face_size_px, self.config.min_face_size_px)

    def test_dark_frame_is_rejected(self) -> None:
        frame = np.full((320, 320, 3), 20, dtype=np.uint8)
        report = self.assessor.assess(frame, make_face((60.0, 60.0, 260.0, 260.0)))
        self.assertFalse(report.passed)
        self.assertTrue(any("dark" in reason for reason in report.reasons))

    def test_small_face_is_rejected(self) -> None:
        report = self.assessor.assess(self.frame, make_face((10.0, 10.0, 50.0, 50.0)))
        self.assertFalse(report.passed)
        self.assertTrue(any("too small" in reason for reason in report.reasons))

    def test_low_detection_score_is_rejected(self) -> None:
        report = self.assessor.assess(
            self.frame, make_face((60.0, 60.0, 260.0, 260.0), det_score=0.2)
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("confidence" in reason for reason in report.reasons))

    def test_missing_landmarks_are_reported(self) -> None:
        face = make_face((60.0, 60.0, 260.0, 260.0))
        face.landmarks = None
        report = self.assessor.assess(self.frame, face)
        self.assertFalse(report.passed)
        self.assertIn("landmarks unavailable", report.reasons)

    def test_face_outside_the_frame_is_rejected(self) -> None:
        report = self.assessor.assess(self.frame, make_face((500.0, 500.0, 700.0, 700.0)))
        self.assertFalse(report.passed)
        self.assertIn("face region is outside the frame", report.reasons)

    def test_turned_head_is_rejected(self) -> None:
        face = make_face((60.0, 60.0, 260.0, 260.0), landmarks=make_landmarks(0.7, 0.0))
        report = self.assessor.assess(self.frame, face)
        self.assertFalse(report.passed)
        self.assertTrue(any("turned" in reason for reason in report.reasons))

    def test_score_is_between_zero_and_one(self) -> None:
        report = self.assessor.assess(self.frame, make_face((60.0, 60.0, 260.0, 260.0)))
        self.assertGreaterEqual(report.score, 0.0)
        self.assertLessEqual(report.score, 1.0)

    def test_summary_mentions_rejection_reasons(self) -> None:
        report = self.assessor.assess(np.zeros((320, 320, 3), dtype=np.uint8), make_face((60.0, 60.0, 260.0, 260.0)))
        self.assertFalse(report.passed)
        self.assertIn("rejected", report.summary())


if __name__ == "__main__":
    unittest.main()
