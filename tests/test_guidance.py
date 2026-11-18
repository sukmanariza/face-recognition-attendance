"""Guidance text for enrollment frames."""

from __future__ import annotations

import unittest

from config import QualityConfig, RegistrationConfig
from src.enrollment_plan import build_plan
from src.face_registration import SampleAttempt, SampleStatus
from src.guidance import (
    GuidanceLevel,
    assess_attempt,
    assess_quality,
    checklist_hints,
)
from tests.support import make_face, make_report

CONFIG = QualityConfig()


def attempt(
    status: SampleStatus = SampleStatus.REJECTED,
    faces: tuple = (),
    quality=None,
    message: str = "",
) -> SampleAttempt:
    return SampleAttempt(
        status=status,
        message=message,
        collected=0,
        required=5,
        prompt="Look at the camera.",
        quality=quality,
        faces=faces,
        stage_title="Front",
    )


class AttemptGuidanceTests(unittest.TestCase):
    def test_complete_state_is_ready(self) -> None:
        result = assess_attempt(attempt(SampleStatus.COMPLETE), None, CONFIG)
        self.assertTrue(result.is_ready)
        self.assertIn("captured", result.headline.lower())

    def test_captured_state_is_ready(self) -> None:
        stage = build_plan(RegistrationConfig())[1]
        result = assess_attempt(attempt(SampleStatus.CAPTURED), stage, CONFIG)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.detail, stage.instruction)

    def test_waiting_state_asks_to_hold_still(self) -> None:
        result = assess_attempt(attempt(SampleStatus.WAITING), None, CONFIG)
        self.assertIn("hold still", result.headline.lower())

    def test_no_face_is_blocked(self) -> None:
        result = assess_attempt(attempt(), None, CONFIG)
        self.assertEqual(result.level, GuidanceLevel.BLOCKED)
        self.assertIn("no face", result.headline.lower())

    def test_multiple_faces_are_blocked(self) -> None:
        result = assess_attempt(attempt(faces=(make_face(), make_face())), None, CONFIG)
        self.assertEqual(result.level, GuidanceLevel.BLOCKED)
        self.assertIn("2 faces", result.headline)

    def test_missing_quality_falls_back_to_the_message(self) -> None:
        result = assess_attempt(attempt(faces=(make_face(),), message="Analysing"), None, CONFIG)
        self.assertEqual(result.detail, "Analysing")

    def test_pose_mismatch_is_reported(self) -> None:
        stage = build_plan(RegistrationConfig())[1]
        result = assess_attempt(
            attempt(faces=(make_face(),), quality=make_report(yaw_ratio=0.35)), stage, CONFIG
        )
        self.assertEqual(result.level, GuidanceLevel.ADJUST)
        self.assertIn("left", result.detail.lower())

    def test_matching_pose_is_ready(self) -> None:
        stage = build_plan(RegistrationConfig())[0]
        result = assess_attempt(
            attempt(faces=(make_face(),), quality=make_report(yaw_ratio=0.0)), stage, CONFIG
        )
        self.assertTrue(result.is_ready)


class QualityGuidanceTests(unittest.TestCase):
    def test_good_image_is_ready(self) -> None:
        self.assertTrue(assess_quality(make_report(), CONFIG).is_ready)

    def test_dark_image_asks_for_light(self) -> None:
        report = make_report(passed=False, reasons=("too dark",), brightness=10.0)
        result = assess_quality(report, CONFIG)
        self.assertEqual(result.level, GuidanceLevel.BLOCKED)
        self.assertIn("dark", result.headline.lower())

    def test_far_face_asks_to_move_closer(self) -> None:
        report = make_report(passed=False, face_size_px=40)
        result = assess_quality(report, CONFIG)
        self.assertIn("far", result.headline.lower())

    def test_checklist_mentions_glasses_and_light(self) -> None:
        text = " ".join(checklist_hints(CONFIG)).lower()
        self.assertIn("glasses", text)
        self.assertIn("light", text)
        self.assertIn("40-60 cm", text)


if __name__ == "__main__":
    unittest.main()
