"""The staged enrollment plan and each pose check."""

from __future__ import annotations

import unittest

from config import RegistrationConfig
from src.enrollment_plan import (
    PoseKind,
    build_plan,
    check_stage,
    progress_label,
)
from tests.support import make_report, make_stage


class PlanTests(unittest.TestCase):
    def test_default_plan_has_five_stages_in_order(self) -> None:
        plan = build_plan(RegistrationConfig())
        self.assertEqual(
            [stage.kind for stage in plan],
            [
                PoseKind.FRONTAL,
                PoseKind.TURN_LEFT,
                PoseKind.TURN_RIGHT,
                PoseKind.TILT,
                PoseKind.CLOSER,
            ],
        )

    def test_plan_is_truncated_to_the_sample_count(self) -> None:
        plan = build_plan(RegistrationConfig(samples=2))
        self.assertEqual([stage.kind for stage in plan], [PoseKind.FRONTAL, PoseKind.TURN_LEFT])

    def test_plan_is_never_shorter_than_one_stage(self) -> None:
        self.assertEqual(len(build_plan(RegistrationConfig(samples=1))), 1)

    def test_stage_titles_and_instructions_are_present(self) -> None:
        for stage in build_plan(RegistrationConfig()):
            self.assertTrue(stage.title)
            self.assertTrue(stage.instruction)

    def test_progress_label_mentions_the_stage(self) -> None:
        stage = build_plan(RegistrationConfig())[0]
        self.assertIn("1 of 5", progress_label(stage, 0, 5))
        self.assertIn(stage.title, progress_label(stage, 0, 5))


class FrontalStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = build_plan(RegistrationConfig())[0]

    def test_frontal_face_is_accepted(self) -> None:
        self.assertTrue(check_stage(self.stage, make_report(yaw_ratio=0.0), None).satisfied)

    def test_slightly_turned_face_is_accepted(self) -> None:
        self.assertTrue(check_stage(self.stage, make_report(yaw_ratio=0.08), None).satisfied)

    def test_turned_face_is_rejected(self) -> None:
        result = check_stage(self.stage, make_report(yaw_ratio=0.3), None)
        self.assertFalse(result.satisfied)
        self.assertIn("frontal", result.detail.lower())

    def test_excessive_roll_is_rejected(self) -> None:
        result = check_stage(self.stage, make_report(roll_deg=22.0), None)
        self.assertFalse(result.satisfied)
        self.assertIn("upright", result.detail.lower())


class TurnStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left, self.right = build_plan(RegistrationConfig())[1:3]

    def test_left_stage_needs_a_negative_yaw(self) -> None:
        self.assertTrue(check_stage(self.left, make_report(yaw_ratio=-0.3), None).satisfied)
        self.assertFalse(check_stage(self.left, make_report(yaw_ratio=0.3), None).satisfied)

    def test_right_stage_needs_a_positive_yaw(self) -> None:
        self.assertTrue(check_stage(self.right, make_report(yaw_ratio=0.3), None).satisfied)
        self.assertFalse(check_stage(self.right, make_report(yaw_ratio=-0.3), None).satisfied)

    def test_wrong_direction_gives_a_direction_hint(self) -> None:
        result = check_stage(self.left, make_report(yaw_ratio=0.3), None)
        self.assertIn("left", result.detail.lower())


class TiltStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = build_plan(RegistrationConfig())[3]

    def test_untilted_head_is_rejected(self) -> None:
        result = check_stage(self.stage, make_report(roll_deg=0.0), None)
        self.assertFalse(result.satisfied)
        self.assertIn("lean", result.detail.lower())

    def test_tilt_inside_the_band_is_accepted(self) -> None:
        self.assertTrue(check_stage(self.stage, make_report(roll_deg=12.0), None).satisfied)

    def test_tilt_beyond_the_band_is_rejected(self) -> None:
        self.assertFalse(check_stage(self.stage, make_report(roll_deg=28.0), None).satisfied)


class CloserStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = build_plan(RegistrationConfig())[4]

    def test_needs_a_larger_face_than_the_reference(self) -> None:
        result = check_stage(self.stage, make_report(face_size_px=180), reference_face_px=200)
        self.assertFalse(result.satisfied)
        self.assertIn("step towards", result.detail.lower())

    def test_large_enough_face_is_accepted(self) -> None:
        result = check_stage(self.stage, make_report(face_size_px=260), reference_face_px=200)
        self.assertTrue(result.satisfied)

    def test_without_a_reference_the_scale_check_is_skipped(self) -> None:
        result = check_stage(self.stage, make_report(face_size_px=100), None)
        self.assertTrue(result.satisfied)


class GenericStageTests(unittest.TestCase):
    def test_negative_yaw_bound_is_honoured(self) -> None:
        stage = make_stage(yaw_min=0.2, yaw_max=1.0)
        self.assertFalse(check_stage(stage, make_report(yaw_ratio=0.0), None).satisfied)
        self.assertTrue(check_stage(stage, make_report(yaw_ratio=0.5), None).satisfied)

    def test_scale_ratio_of_zero_ignores_the_face_size(self) -> None:
        stage = make_stage(min_scale_ratio=0.0)
        self.assertTrue(check_stage(stage, make_report(face_size_px=40), 400).satisfied)


if __name__ == "__main__":
    unittest.main()
