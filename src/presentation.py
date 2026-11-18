"""Presentation check: is the face in front of the camera a real (3D) face?

The only cue used here is geometric, and it is a real measurement rather than a
guess: a face on a flat photo or a screen is a **plane**, while a real face has the
nose and cheeks standing out of the eye/mouth plane. When the head moves, a plane
can be described by a single homography with almost no residual, while a real face
leaves a residual concentrated at the nose.

So the test is: fit a homography between two frames of the same face and compare
the nose residual with the residual of the other landmarks. Ratios near 1 mean the
surface behaved like a plane; ratios well above 1 mean it had depth.

What this catches and what it does not:

* a printed photo or a still image on a screen, moved or turned in the hand: the
  residual stays planar, so it is flagged;
* a real face: the nose residual grows with head yaw, so it is confirmed;
* a video replay of a real person on a screen, a curved or bent print, or a 3D
  mask: the parallax looks real, so it can pass. Detection is not certification.

With no movement there is no baseline and therefore no evidence, which is why the
third state exists instead of a guess. Callers must treat `UNCERTAIN` as unknown,
never as "live".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from config import LivenessConfig

from .face_detector import DetectedFace

NOSE_INDEX = 2
LEFT_EYE_INDEX = 0
RIGHT_EYE_INDEX = 1
_MIN_INTEROCULAR_PX = 6.0
_MAX_OBSERVATIONS = 24
_MAX_PAIRS = 4
# Below this the residuals are dominated by landmark jitter, so the ratio is noise.
_MIN_RESIDUAL_FLOOR = 0.002


class SpoofLevel(StrEnum):
    """What the presentation check could establish."""

    GENUINE = "genuine"
    UNCERTAIN = "uncertain"
    SUSPECTED = "suspected"


@dataclass(frozen=True)
class PlanarEvidence:
    """One frame pair's depth evidence, scaled by the inter-ocular distance.

    ``noise_level`` is the residual of the four coplanar landmarks, which measures
    how much the detector jitters. ``nose_ratio`` is kept for diagnostics only: with
    four samples it swings between 1 and 3 on noise alone.
    """

    nose_residual: float
    noise_level: float
    motion: float
    nose_ratio: float


@dataclass(frozen=True)
class PresentationVerdict:
    """Result of the presentation check for one face."""

    level: SpoofLevel
    confidence: float
    detail: str
    observations: int
    planar_pairs: int = 0
    solid_pairs: int = 0

    @property
    def is_genuine(self) -> bool:
        return self.level is SpoofLevel.GENUINE

    @property
    def is_suspected(self) -> bool:
        return self.level is SpoofLevel.SUSPECTED

    @property
    def is_live(self) -> bool | None:
        """Tri-state mapping used by the older liveness contract."""
        if self.level is SpoofLevel.GENUINE:
            return True
        if self.level is SpoofLevel.SUSPECTED:
            return False
        return None

    @property
    def label(self) -> str:
        if self.level is SpoofLevel.GENUINE:
            return "Real face (3D)"
        if self.level is SpoofLevel.SUSPECTED:
            return "Photo or screen (2D)"
        return "Not verified"

    @property
    def short_label(self) -> str:
        if self.level is SpoofLevel.GENUINE:
            return "3D"
        if self.level is SpoofLevel.SUSPECTED:
            return "2D"
        return "?"


@dataclass(frozen=True)
class FaceObservation:
    """Landmarks of one face in one frame."""

    points: np.ndarray
    interocular: float
    timestamp: float


def interocular_distance(points: np.ndarray) -> float:
    """Pixel distance between the eye centres."""
    delta = np.asarray(points, dtype=np.float64)[RIGHT_EYE_INDEX] - np.asarray(
        points, dtype=np.float64
    )[LEFT_EYE_INDEX]
    return float(np.linalg.norm(delta))


def planar_evidence(
    first: np.ndarray, second: np.ndarray, min_motion: float
) -> PlanarEvidence | None:
    """Compare two landmark sets of the same face.

    Returns ``None`` when the pair cannot answer the question: too little movement
    between the frames, or a degenerate configuration.
    """
    source = np.asarray(first, dtype=np.float32).reshape(5, 2)
    target = np.asarray(second, dtype=np.float32).reshape(5, 2)
    scale = interocular_distance(source)
    if scale < _MIN_INTEROCULAR_PX:
        return None

    motion = float(np.mean(np.linalg.norm(target - source, axis=1))) / scale
    if motion < min_motion:
        return None

    matrix, _mask = cv2.findHomography(source, target, 0)
    if matrix is None:
        return None
    projected = cv2.perspectiveTransform(source.reshape(1, -1, 2), matrix).reshape(-1, 2)
    residual = np.linalg.norm(target - projected, axis=1) / scale

    others = np.delete(residual, NOSE_INDEX)
    other_median = float(np.median(others))
    nose = float(residual[NOSE_INDEX])
    return PlanarEvidence(
        nose_residual=nose,
        noise_level=other_median,
        motion=motion,
        nose_ratio=nose / max(other_median, _MIN_RESIDUAL_FLOOR),
    )


class PresentationGuard:
    """Accumulate landmarks over time and decide whether the face has depth."""

    name = "planar-parallax"

    def __init__(self, config: LivenessConfig) -> None:
        self._config = config
        self._buffer: deque[FaceObservation] = deque(maxlen=_MAX_OBSERVATIONS)
        self._verdict = self._uncertain("Waiting for the first frame.", 0)

    # -- thresholds derived from one sensitivity dial ---------------------
    @property
    def _sensitivity(self) -> float:
        return min(1.0, max(0.0, self._config.sensitivity))

    @property
    def min_motion(self) -> float:
        """Head movement needed before a frame pair counts, relative to eye distance."""
        return 0.028 - 0.010 * self._sensitivity

    @property
    def solid_floor(self) -> float:
        """Nose residual that cannot be explained by a flat surface."""
        return 0.020

    @property
    def noise_factor(self) -> float:
        """How far the nose must stand out above the detector's own jitter."""
        return 2.9 - 0.6 * self._sensitivity

    @property
    def planar_ceiling(self) -> float:
        """Nose residual small enough to be indistinguishable from the flat plane."""
        return 0.012

    def _is_solid(self, evidence: PlanarEvidence) -> bool:
        limit = max(self.solid_floor, self.noise_factor * evidence.noise_level)
        return evidence.nose_residual >= limit

    def _is_planar(self, evidence: PlanarEvidence) -> bool:
        limit = max(self.planar_ceiling, 1.5 * evidence.noise_level)
        return evidence.nose_residual <= limit

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        self._buffer.clear()
        self._verdict = self._uncertain("Waiting for the first frame.", 0)

    @property
    def verdict(self) -> PresentationVerdict:
        return self._verdict

    @property
    def observation_count(self) -> int:
        return len(self._buffer)

    def observe(self, face: DetectedFace, timestamp: float) -> PresentationVerdict:
        """Record one frame and return the current verdict."""
        self._record(face, timestamp)
        self._verdict = self._decide()
        return self._verdict

    def _record(self, face: DetectedFace, timestamp: float) -> None:
        if face.landmarks is None:
            return
        points = np.asarray(face.landmarks, dtype=np.float32)
        if points.shape != (5, 2):
            return
        scale = interocular_distance(points)
        if scale < _MIN_INTEROCULAR_PX:
            return
        self._buffer.append(FaceObservation(points.copy(), scale, timestamp))

    # -- decision --------------------------------------------------------
    def _decide(self) -> PresentationVerdict:
        count = len(self._buffer)
        if count < max(2, self._config.min_observations):
            return self._uncertain(
                f"Collecting movement ({count}/{self._config.min_observations}).", count
            )

        pairs = self._evidence_pairs()
        if not pairs:
            return self._uncertain(
                "No usable movement yet. Keep the face in view and move the head slightly.",
                count,
            )

        planar = sum(1 for pair in pairs if self._is_planar(pair))
        solid = sum(1 for pair in pairs if self._is_solid(pair))

        if solid and not planar:
            return PresentationVerdict(
                level=SpoofLevel.GENUINE,
                confidence=round(solid / len(pairs), 3),
                detail="The face moved with depth, like a real head.",
                observations=count,
                planar_pairs=planar,
                solid_pairs=solid,
            )
        if planar and not solid:
            return PresentationVerdict(
                level=SpoofLevel.SUSPECTED,
                confidence=round(planar / len(pairs), 3),
                detail="The face moved as a flat surface, so it is a photo or a screen.",
                observations=count,
                planar_pairs=planar,
                solid_pairs=solid,
            )
        return PresentationVerdict(
            level=SpoofLevel.UNCERTAIN,
            confidence=0.0,
            detail="Signals disagree: depth was seen in some frames and not in others.",
            observations=count,
            planar_pairs=planar,
            solid_pairs=solid,
        )

    def _evidence_pairs(self) -> list[PlanarEvidence]:
        """Compare the newest observation against a few older ones."""
        newest = self._buffer[-1]
        older = list(self._buffer)[:-1]
        if not older:
            return []

        picks = min(_MAX_PAIRS, len(older))
        step = max(1, len(older) // picks)
        chosen = older[::step][:picks]

        evidence: list[PlanarEvidence] = []
        for observation in chosen:
            pair = planar_evidence(observation.points, newest.points, self.min_motion)
            if pair is not None:
                evidence.append(pair)
        return evidence

    @staticmethod
    def _uncertain(detail: str, count: int) -> PresentationVerdict:
        return PresentationVerdict(
            level=SpoofLevel.UNCERTAIN,
            confidence=0.0,
            detail=detail,
            observations=count,
        )

    # -- helpers for the UI ---------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._config.mode != "off"

    def describe(self) -> str:
        return (
            f"planar parallax, sensitivity {self._sensitivity:.2f}, "
            f"nose residual floor {self.solid_floor:.3f}, "
            f"flat ceiling {self.planar_ceiling:.3f}, "
            f"min motion {self.min_motion:.1%} of eye distance"
        )
