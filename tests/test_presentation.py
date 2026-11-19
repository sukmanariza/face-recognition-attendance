"""Presentation check: does the face in front of the camera have depth?

The sequences here are synthetic but not arbitrary: a five-point face model is
projected through a pinhole camera at several head yaws, once with the nose on the
eye/mouth plane (a photo) and once with it standing out (a real face). Landmark
jitter is added because real detectors never return exact positions.
"""

from __future__ import annotations

import unittest
from math import cos, radians, sin

import numpy as np

from config import LivenessConfig
from src.face_detector import DetectedFace
from src.liveness import (
    DisabledLivenessDetector,
    LivenessResult,
    PresentationLivenessDetector,
    build_liveness_detector,
)
from src.presentation import (
    PresentationGuard,
    PresentationVerdict,
    SpoofLevel,
    interocular_distance,
    planar_evidence,
)

DISTANCE = 300.0  # camera to face, model units
FOCAL = 600.0
CENTER = (320.0, 240.0)


def face_model(nose_depth: float) -> np.ndarray:
    """Five landmarks in model space; z points towards the camera."""
    return np.array(
        [
            [-40.0, -10.0, 0.0],  # left eye
            [40.0, -10.0, 0.0],  # right eye
            [0.0, 18.0, nose_depth],  # nose
            [-22.0, 40.0, 0.0],
            [22.0, 40.0, 0.0],
        ],
        dtype=np.float64,
    )


def project(points: np.ndarray, yaw_deg: float) -> np.ndarray:
    theta = radians(yaw_deg)
    rotation = np.array([[cos(theta), 0.0, sin(theta)], [0.0, 1.0, 0.0], [-sin(theta), 0.0, cos(theta)]])
    rotated = points @ rotation.T
    depth = DISTANCE - rotated[:, 2]
    x = FOCAL * rotated[:, 0] / depth + CENTER[0]
    y = FOCAL * rotated[:, 1] / depth + CENTER[1]
    return np.stack([x, y], axis=1).astype(np.float32)


def jitter(points: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return (points + generator.normal(0.0, sigma, points.shape)).astype(np.float32)


def sequence(nose_depth: float, yaws: tuple[float, ...], sigma: float = 0.6) -> list[np.ndarray]:
    model = face_model(nose_depth)
    return [jitter(project(model, yaw), sigma, index) for index, yaw in enumerate(yaws)]


def as_face(points: np.ndarray) -> DetectedFace:
    return DetectedFace(
        bbox=np.array([300.0, 180.0, 500.0, 400.0], dtype=np.float32),
        det_score=0.95,
        landmarks=points,
    )


def guard_config(**overrides) -> LivenessConfig:
    base = {"mode": "advisory", "min_observations": 4, "sensitivity": 0.5}
    base.update(overrides)
    return LivenessConfig(**base)


def run_sequence(
    points: list[np.ndarray],
    config: LivenessConfig | None = None,
    *,
    seconds: float = 5.0,
    interval: float = 0.1,
) -> PresentationVerdict:
    """Replay the pose sequence like a live session, for a few seconds.

    The guard only decides every ``decision_interval_seconds``, so a handful of
    frames is not enough to see a verdict.
    """
    guard = PresentationGuard(config or guard_config())
    steps = max(1, round(seconds / interval))
    verdict = guard.verdict
    for step in range(steps):
        verdict = guard.observe(as_face(points[step % len(points)]), timestamp=step * interval)
    return verdict


class GeometryTests(unittest.TestCase):
    def test_interocular_distance_is_measured_between_the_eyes(self) -> None:
        points = np.array(
            [[0.0, 0.0], [80.0, 0.0], [40.0, 20.0], [20.0, 40.0], [60.0, 40.0]],
            dtype=np.float32,
        )
        self.assertAlmostEqual(interocular_distance(points), 80.0)

    def test_pairs_without_movement_are_rejected(self) -> None:
        points = project(face_model(20.0), 0.0)
        self.assertIsNone(planar_evidence(points, points, min_motion=0.05))

    def test_a_flat_surface_has_no_nose_residual(self) -> None:
        pair = planar_evidence(project(face_model(0.0), 0.0), project(face_model(0.0), 30.0), 0.02)
        assert pair is not None
        self.assertLess(pair.nose_residual, 0.012)
        self.assertLessEqual(pair.nose_residual, 1.5 * pair.noise_level)

    def test_a_nose_that_stands_out_leaves_a_residual(self) -> None:
        pair = planar_evidence(project(face_model(20.0), 0.0), project(face_model(20.0), 30.0), 0.02)
        assert pair is not None
        self.assertGreater(pair.nose_residual, 0.020)
        self.assertGreater(pair.nose_residual, 2.0 * pair.noise_level)


class GuardTests(unittest.TestCase):
    YAWS = (0.0, 8.0, 16.0, 24.0, 32.0)

    def test_a_flat_photo_is_suspected(self) -> None:
        verdict = run_sequence(sequence(0.0, self.YAWS))
        self.assertEqual(verdict.level, SpoofLevel.SUSPECTED)
        self.assertFalse(verdict.is_live)
        self.assertEqual(verdict.label, "Photo or screen (2D)")
        self.assertGreater(verdict.planar_pairs, 0)

    def test_a_real_face_is_confirmed(self) -> None:
        verdict = run_sequence(sequence(20.0, self.YAWS))
        self.assertEqual(verdict.level, SpoofLevel.GENUINE)
        self.assertTrue(verdict.is_live)
        self.assertEqual(verdict.label, "Real face (3D)")
        self.assertGreater(verdict.solid_pairs, 0)

    def test_too_few_frames_stay_undecided(self) -> None:
        verdict = run_sequence(sequence(20.0, (0.0, 20.0)))
        self.assertEqual(verdict.level, SpoofLevel.UNCERTAIN)
        self.assertIsNone(verdict.is_live)
        self.assertEqual(verdict.label, "Not verified")

    def test_a_still_face_gives_no_evidence(self) -> None:
        still = [project(face_model(20.0), 0.0)] * 6
        verdict = run_sequence(still)
        self.assertEqual(verdict.level, SpoofLevel.UNCERTAIN)
        self.assertIn("movement", verdict.detail.lower())

    def test_reset_clears_the_buffer(self) -> None:
        guard = PresentationGuard(guard_config())
        for index, frame_points in enumerate(sequence(20.0, self.YAWS)):
            guard.observe(as_face(frame_points), timestamp=index * 0.1)
        guard.reset()
        self.assertEqual(guard.observation_count, 0)
        self.assertEqual(guard.verdict.level, SpoofLevel.UNCERTAIN)

    def test_sensitivity_moves_the_thresholds(self) -> None:
        strict = PresentationGuard(guard_config(sensitivity=0.0))
        loose = PresentationGuard(guard_config(sensitivity=1.0))
        self.assertGreater(strict.min_motion, loose.min_motion)
        self.assertGreater(strict.noise_factor, loose.noise_factor)
        self.assertEqual(strict.solid_floor, loose.solid_floor)

    def test_observations_without_landmarks_are_ignored(self) -> None:
        guard = PresentationGuard(guard_config())
        face = as_face(project(face_model(20.0), 0.0))
        face.landmarks = None
        guard.observe(face, timestamp=0.0)
        self.assertEqual(guard.observation_count, 0)

    def test_describe_mentions_the_measure(self) -> None:
        self.assertIn("parallax", PresentationGuard(guard_config()).describe())


class StabilityTests(unittest.TestCase):
    """The label has to hold still: it is read by a person watching the screen."""

    # A head that keeps turning, never repeating the same pose within a window.
    SWEEP = tuple(1.5 * index for index in range(24))

    @staticmethod
    def _levels(guard: PresentationGuard, points: list[np.ndarray], steps: int) -> list[SpoofLevel]:
        levels: list[SpoofLevel] = []
        for step in range(steps):
            verdict = guard.observe(as_face(points[step % len(points)]), timestamp=step * 0.1)
            levels.append(verdict.level)
        return levels

    def test_the_label_holds_for_a_whole_session(self) -> None:
        for depth, expected in ((0.0, SpoofLevel.SUSPECTED), (20.0, SpoofLevel.GENUINE)):
            with self.subTest(depth=depth):
                guard = PresentationGuard(guard_config())
                levels = self._levels(guard, sequence(depth, self.SWEEP), 200)

                changes = [
                    index
                    for index in range(1, len(levels))
                    if levels[index] is not levels[index - 1]
                ]
                self.assertEqual(len(changes), 1, f"levels changed at {changes}")
                self.assertEqual(levels[-1], expected)

    def test_the_decision_waits_for_the_interval(self) -> None:
        guard = PresentationGuard(guard_config(decision_interval_seconds=2.0))
        points = sequence(20.0, self.SWEEP)

        raised = None
        for step in range(80):
            verdict = guard.observe(as_face(points[step % len(points)]), timestamp=step * 0.1)
            if verdict.level is SpoofLevel.GENUINE:
                raised = step
                break

        self.assertIsNotNone(raised, "the verdict never became genuine")
        assert raised is not None
        # Decisions land at 0 s, 2 s and 4 s, never on every frame.
        self.assertGreaterEqual(raised * 0.1, 1.9)

    def test_one_contradicting_window_does_not_flip_the_label(self) -> None:
        guard = PresentationGuard(guard_config(stability_windows=3))
        real = sequence(20.0, self.SWEEP)
        flat = sequence(0.0, self.SWEEP)

        for step in range(100):
            guard.observe(as_face(real[step % len(real)]), timestamp=step * 0.1)
        self.assertEqual(guard.verdict.level, SpoofLevel.GENUINE)

        # One window of photo evidence, which is not enough to change the label.
        for step in range(100, 115):
            guard.observe(as_face(flat[step % len(flat)]), timestamp=step * 0.1)

        self.assertEqual(guard.verdict.level, SpoofLevel.GENUINE)

    def test_a_pause_forgets_the_old_observations(self) -> None:
        guard = PresentationGuard(guard_config(observation_window_seconds=1.0))
        points = sequence(20.0, self.SWEEP)
        for step in range(20):
            guard.observe(as_face(points[step]), timestamp=step * 0.1)
        self.assertGreater(guard.observation_count, 5)

        # The person left the frame for ten seconds.
        guard.observe(as_face(points[0]), timestamp=10.0)

        self.assertEqual(guard.observation_count, 1)

    def test_reset_also_clears_the_pending_change(self) -> None:
        guard = PresentationGuard(guard_config())
        points = sequence(20.0, self.SWEEP)
        for step in range(40):
            guard.observe(as_face(points[step % len(points)]), timestamp=step * 0.1)

        guard.reset()
        verdict = guard.observe(as_face(points[0]), timestamp=100.0)

        self.assertEqual(verdict.level, SpoofLevel.UNCERTAIN)
        self.assertIn("Collecting", verdict.detail)


class DetectorTests(unittest.TestCase):
    def test_off_mode_gives_the_disabled_detector(self) -> None:
        detector = build_liveness_detector(LivenessConfig(mode="off"))
        self.assertIsInstance(detector, DisabledLivenessDetector)
        result = detector.check(np.zeros((4, 4, 3), dtype=np.uint8), as_face(project(face_model(0.0), 0.0)))
        self.assertIsNone(result.is_live)
        self.assertIsNone(result.verdict)

    def test_advisory_mode_measures(self) -> None:
        detector = build_liveness_detector(LivenessConfig(mode="advisory"))
        self.assertIsInstance(detector, PresentationLivenessDetector)
        result = detector.check(
            np.zeros((4, 4, 3), dtype=np.uint8), as_face(project(face_model(0.0), 0.0)), 0.0
        )
        self.assertIsInstance(result, LivenessResult)
        self.assertEqual(result.label, "Not verified")

    def test_reset_clears_the_detector(self) -> None:
        detector = build_liveness_detector(LivenessConfig(mode="advisory"))
        assert isinstance(detector, PresentationLivenessDetector)
        for index, frame_points in enumerate(sequence(0.0, (0.0, 10.0, 20.0, 30.0))):
            detector.check(np.zeros((4, 4, 3), dtype=np.uint8), as_face(frame_points), index * 0.1)
        detector.reset()
        self.assertEqual(detector.guard.observation_count, 0)


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = LivenessConfig()
        self.assertEqual(config.mode, "advisory")
        self.assertTrue(config.enabled)
        self.assertFalse(config.enforcing)

    def test_off_is_not_enabled(self) -> None:
        self.assertFalse(LivenessConfig(mode="off").enabled)

    def test_enforce_is_enforcing(self) -> None:
        self.assertTrue(LivenessConfig(mode="enforce").enforcing)

    def test_invalid_mode_is_rejected(self) -> None:
        from config import ConfigError, build_config

        with self.assertRaises(ConfigError):
            build_config({"liveness": {"mode": "block"}})

    def test_observation_count_is_bounded(self) -> None:
        from config import ConfigError, build_config

        with self.assertRaises(ConfigError):
            build_config({"liveness": {"min_observations": 1}})

    def test_sensitivity_is_bounded(self) -> None:
        from config import ConfigError, build_config

        with self.assertRaises(ConfigError):
            build_config({"liveness": {"sensitivity": 2.0}})


if __name__ == "__main__":
    unittest.main()
