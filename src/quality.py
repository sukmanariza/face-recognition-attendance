"""Face quality assessment used to accept or reject a registration sample.

Rejecting poor samples is what keeps recognition accurate: an embedding taken
from a blurred, dark, or turned-away face produces misleading similarity scores
for as long as that identity exists.

Thresholds depend on camera, lighting, and sitting distance. Defaults are a
starting point, not universal constants; TECHNICAL.md explains how to calibrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import QualityConfig

from .face_detector import DetectedFace

# Blur is measured on a fixed-size crop so moving closer does not change the score.
_BLUR_ANALYSIS_SIZE = 160
_CROP_MARGIN = 0.10


@dataclass(frozen=True)
class QualityReport:
    """Measurements for one face in one frame."""

    passed: bool
    reasons: tuple[str, ...]
    face_size_px: int
    det_score: float
    blur_score: float
    brightness: float
    contrast: float
    yaw_ratio: float
    roll_deg: float

    @property
    def score(self) -> float:
        """0-1 summary stored as sample metadata; decisions use ``passed``."""
        parts = [self.det_score, self.contrast / 255.0, min(1.0, self.face_size_px / 320.0)]
        return float(np.clip(np.mean(parts), 0.0, 1.0))

    def summary(self) -> str:
        if self.passed:
            return (
                f"good ({self.face_size_px} px, sharpness {self.blur_score:.0f}, "
                f"light {self.brightness:.0f}, tilt {self.roll_deg:.1f} deg)"
            )
        return "rejected: " + "; ".join(self.reasons)


def estimate_pose(landmarks: np.ndarray) -> tuple[float, float]:
    """Estimate head yaw (unitless ratio) and roll (degrees) from 5 landmarks.

    Landmark order is left eye, right eye, nose, left mouth corner, right mouth
    corner in image coordinates. ``yaw_ratio`` is 0 for a frontal face and grows
    towards 1 as the head turns; it stays unitless because converting it to
    degrees needs a 3D face model, and a quality gate only needs monotonicity.
    ``roll_deg`` is exact since it depends on two 2D eye points only.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError(f"Landmarks must be shaped (5, 2), got {points.shape}.")

    left_eye, right_eye, nose = points[0], points[1], points[2]
    eye_delta = right_eye - left_eye
    roll_deg = float(np.degrees(np.arctan2(float(eye_delta[1]), float(eye_delta[0]))))

    distance_left = float(np.linalg.norm(nose - left_eye))
    distance_right = float(np.linalg.norm(nose - right_eye))
    total = distance_left + distance_right
    yaw_ratio = 0.0 if total < 1e-6 else (distance_right - distance_left) / total
    return float(np.clip(yaw_ratio, -1.0, 1.0)), roll_deg


class FaceQualityAssessor:
    """Measure and grade the quality of a detected face."""

    def __init__(self, config: QualityConfig) -> None:
        self._config = config

    @property
    def config(self) -> QualityConfig:
        return self._config

    def assess(self, frame: np.ndarray, face: DetectedFace) -> QualityReport:
        config = self._config
        reasons: list[str] = []

        face_size = round(face.min_side)
        if face_size < config.min_face_size_px:
            reasons.append(f"face too small ({face_size} px, needs {config.min_face_size_px})")
        if face.det_score < config.min_det_score:
            reasons.append(
                f"low detection confidence ({face.det_score:.2f}, needs {config.min_det_score:.2f})"
            )

        grey = self._face_grey(frame, face)
        if grey is None:
            return QualityReport(
                passed=False,
                reasons=("face region is outside the frame",),
                face_size_px=face_size,
                det_score=float(face.det_score),
                blur_score=0.0,
                brightness=0.0,
                contrast=0.0,
                yaw_ratio=0.0,
                roll_deg=0.0,
            )

        blur_score = self._blur_score(grey)
        brightness = float(np.mean(grey))
        contrast = float(np.std(grey))

        if blur_score < config.blur_threshold:
            reasons.append(f"blurry (sharpness {blur_score:.0f}, needs {config.blur_threshold:.0f})")
        if brightness < config.brightness_min:
            reasons.append(f"too dark ({brightness:.0f}, needs {config.brightness_min:.0f})")
        if brightness > config.brightness_max:
            reasons.append(f"too bright ({brightness:.0f}, limit {config.brightness_max:.0f})")
        if contrast < config.contrast_min:
            reasons.append(f"low contrast ({contrast:.0f}, needs {config.contrast_min:.0f})")

        yaw_ratio, roll_deg = 0.0, 0.0
        if face.landmarks is None:
            reasons.append("landmarks unavailable")
        else:
            yaw_ratio, roll_deg = estimate_pose(face.landmarks)
            if abs(yaw_ratio) > config.max_yaw_ratio:
                reasons.append(
                    f"head turned too far ({abs(yaw_ratio):.2f}, limit {config.max_yaw_ratio:.2f})"
                )
            if abs(roll_deg) > config.max_roll_deg:
                reasons.append(
                    f"head tilted too far ({abs(roll_deg):.1f} deg, "
                    f"limit {config.max_roll_deg:.1f})"
                )

        return QualityReport(
            passed=not reasons,
            reasons=tuple(reasons),
            face_size_px=face_size,
            det_score=float(face.det_score),
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            yaw_ratio=yaw_ratio,
            roll_deg=roll_deg,
        )

    @staticmethod
    def _face_grey(frame: np.ndarray, face: DetectedFace) -> np.ndarray | None:
        """Crop the face with a small margin, convert to grey, and normalise size."""
        x1, y1, x2, y2 = face.integer_bbox(frame.shape)
        width, height = x2 - x1, y2 - y1
        if width < 4 or height < 4:
            return None
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, x1 - round(width * _CROP_MARGIN))
        y1 = max(0, y1 - round(height * _CROP_MARGIN))
        x2 = min(frame_width, x2 + round(width * _CROP_MARGIN))
        y2 = min(frame_height, y2 + round(height * _CROP_MARGIN))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return cv2.resize(
            grey, (_BLUR_ANALYSIS_SIZE, _BLUR_ANALYSIS_SIZE), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _blur_score(grey: np.ndarray) -> float:
        return float(cv2.Laplacian(grey, cv2.CV_64F).var())
