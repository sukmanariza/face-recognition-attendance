"""Shared test doubles, so tests never download a model or open a camera."""

from __future__ import annotations

import numpy as np

from src.enrollment_plan import PoseStage
from src.face_detector import DetectedFace
from src.liveness import LivenessResult
from src.presentation import PresentationVerdict, SpoofLevel
from src.quality import QualityReport

EMBEDDING_DIM = 8


def make_face(
    bbox: tuple[float, float, float, float] = (40.0, 40.0, 200.0, 200.0),
    det_score: float = 0.92,
    landmarks: np.ndarray | None = None,
) -> DetectedFace:
    if landmarks is None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        landmarks = np.array(
            [
                [cx - 20.0, cy - 20.0],
                [cx + 20.0, cy - 20.0],
                [cx, cy],
                [cx - 14.0, cy + 22.0],
                [cx + 14.0, cy + 22.0],
            ],
            dtype=np.float32,
        )
    return DetectedFace(
        bbox=np.asarray(bbox, dtype=np.float32),
        det_score=det_score,
        landmarks=landmarks,
    )


def make_landmarks(yaw_ratio: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Landmarks that measure back to the requested yaw ratio and roll angle.

    With the nose placed on the line between the eyes, ``estimate_pose`` returns
    ``(1 - 2t)`` for a nose at fraction ``t``, so the pattern can be solved for the
    ratio exactly instead of guessed. Rotation preserves distances, so the ratio is
    unaffected by the roll.
    """
    clipped = float(np.clip(yaw_ratio, -1.0, 1.0))
    fraction = (1.0 - clipped) / 2.0
    centre_x, centre_y = 100.0, 100.0
    half = 20.0
    points = np.array(
        [
            [centre_x - half, centre_y],
            [centre_x + half, centre_y],
            [centre_x - half + 2.0 * half * fraction, centre_y],
            [centre_x - 14.0, centre_y + 22.0],
            [centre_x + 14.0, centre_y + 22.0],
        ],
        dtype=np.float32,
    )
    angle = np.radians(roll_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32
    )
    centre = np.array([centre_x, centre_y], dtype=np.float32)
    points[:3] = (points[:3] - centre) @ rotation.T + centre
    return points


def make_report(
    passed: bool = True,
    yaw_ratio: float = 0.0,
    roll_deg: float = 0.0,
    face_size_px: int = 200,
    reasons: tuple[str, ...] = (),
    brightness: float = 120.0,
    blur_score: float = 140.0,
    contrast: float = 60.0,
    det_score: float = 0.92,
) -> QualityReport:
    return QualityReport(
        passed=passed,
        reasons=reasons,
        face_size_px=face_size_px,
        det_score=det_score,
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        yaw_ratio=yaw_ratio,
        roll_deg=roll_deg,
    )


def make_stage(
    yaw_min: float = -1.0,
    yaw_max: float = 1.0,
    roll_min: float = -90.0,
    roll_max: float = 90.0,
    min_scale_ratio: float = 0.0,
) -> PoseStage:
    from src.enrollment_plan import PoseKind

    return PoseStage(
        kind=PoseKind.FRONTAL,
        title="Test",
        instruction="Look at the camera.",
        yaw_min=yaw_min,
        yaw_max=yaw_max,
        roll_min=roll_min,
        roll_max=roll_max,
        min_scale_ratio=min_scale_ratio,
    )


def unit_vector(seed: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)

class StubLiveness:
    """Reports a fixed presentation verdict, as the real detector would."""

    name = "stub"

    def __init__(self, level: SpoofLevel) -> None:
        self._level = level
        self.calls = 0
        self.resets = 0

    def check(self, frame, face, timestamp: float = 0.0):
        self.calls += 1
        verdict = PresentationVerdict(
            level=self._level,
            confidence=1.0,
            detail="stub verdict",
            observations=8,
        )
        return LivenessResult(is_live=verdict.is_live, score=1.0, reason="stub", verdict=verdict)

    def reset(self) -> None:
        self.resets += 1

def textured_frame(brightness: int = 124, block: int = 40, size: int = 320) -> np.ndarray:
    """Checkerboard frame: high local contrast that survives downscaling.

    Pure noise loses its contrast when the crop is resized, and a linear gradient
    has no edges, so neither can pass the sharpness and contrast checks. Blocks do.
    """
    coordinates = np.arange(size)
    grid = (coordinates[:, None] // block + coordinates[None, :] // block) % 2
    values = np.where(grid == 1, 188, 60).astype(np.uint8)
    del brightness
    return np.dstack([values, values, values])
