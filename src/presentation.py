"""Presentation check: is the face in front of the camera a real (3D) face?

The only cue used here is geometric, and it is a real measurement rather than a
guess: a face on a flat photo or a screen is a **plane**, while a real face has the
nose and cheeks standing out of the eye/mouth plane. When the head moves, a plane
can be described by a single homography with almost no residual, while a real face
leaves a residual concentrated at the nose.

So the test is: fit a homography between two frames of the same face, using only the
four landmarks that are coplanar by construction (both eyes, both mouth corners), and
measure how far the nose lands from where that plane predicts it. The fit is exact on
those four points, so nothing of the depth is absorbed into it and the nose residual
is the whole signal. Values near the landmark jitter mean the surface behaved like a
plane; values well above it mean the face had depth.

One frame pair is noisy, and a label that changes every second is worse than no label.
Three mechanisms keep the published answer steady:

* the decision is recomputed every ``decision_interval_seconds``, not on every frame;
  landmarks are still collected in between;
* it is taken from the **median over every usable pair in the window**, not from a
  single pair and not from a vote of per-pair verdicts;
* a new level must win ``stability_windows`` decisions in a row before it is
  published, so one doubtful window cannot flip what the user reads.

What this catches and what it does not:

* a printed photo or a still image on a screen, moved or turned in the hand: the
  residual stays at the jitter level, so it is flagged;
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
from statistics import median

import cv2
import numpy as np

from config import LivenessConfig

from .face_detector import DetectedFace

NOSE_INDEX = 2
LEFT_EYE_INDEX = 0
RIGHT_EYE_INDEX = 1
# Eyes and mouth corners are coplanar; the nose is not. Using all five lets the fit
# absorb the depth the test is trying to measure.
COPLANAR_INDICES = (0, 1, 3, 4)
_MIN_INTEROCULAR_PX = 6.0
_MAX_OBSERVATIONS = 24
_MAX_PAIRS = 8


class SpoofLevel(StrEnum):
    """What the presentation check could establish."""

    GENUINE = "genuine"
    UNCERTAIN = "uncertain"
    SUSPECTED = "suspected"


@dataclass(frozen=True)
class PlanarEvidence:
    """One frame pair's depth evidence, scaled by the inter-ocular distance.

    ``noise_level`` is the residual of the four coplanar landmarks, which measures
    how much the detector jitters.
    """

    nose_residual: float
    noise_level: float
    motion: float


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

    matrix, _mask = cv2.findHomography(source[list(COPLANAR_INDICES)], target[list(COPLANAR_INDICES)], 0)
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
    )


class PresentationGuard:
    """Accumulate landmarks over time and decide whether the face has depth.

    The measurement itself is noisy frame to frame: a photo and a real face differ by
    a ratio computed from five landmarks, so a single window can swing either way.
    Two mechanisms keep the published label steady:

    * the decision is recomputed every ``decision_interval_seconds``, not on every
      frame, while landmarks keep being collected in between;
    * a new level has to win ``stability_windows`` decisions in a row before it is
      published, so one doubtful window cannot flip the label.
    """

    name = "planar-parallax"

    def __init__(self, config: LivenessConfig) -> None:
        self._config = config
        self._buffer: deque[FaceObservation] = deque(maxlen=_MAX_OBSERVATIONS)
        self._verdict = self._uncertain("Waiting for the first frame.", 0)
        self._decided_at: float | None = None
        self._pending_level: SpoofLevel | None = None
        self._pending_count = 0

    # -- thresholds derived from one sensitivity dial ---------------------
    @property
    def _sensitivity(self) -> float:
        return min(1.0, max(0.0, self._config.sensitivity))

    @property
    def decision_interval(self) -> float:
        return max(0.2, self._config.decision_interval_seconds)

    @property
    def stability_windows(self) -> int:
        return max(1, self._config.stability_windows)

    @property
    def min_motion(self) -> float:
        """Head movement needed before a frame pair counts, relative to eye distance."""
        return 0.028 - 0.010 * self._sensitivity

    @property
    def solid_floor(self) -> float:
        """Nose residual that cannot be explained by a flat surface."""
        return 0.022

    @property
    def noise_factor(self) -> float:
        """How far the nose must stand out above the detector's own jitter."""
        return 2.9 - 0.6 * self._sensitivity

    @property
    def planar_ceiling(self) -> float:
        """Nose residual small enough to be a plane seen through landmark jitter.

        The homography is fitted on the four coplanar landmarks, so it reproduces
        them exactly and the nose residual is left holding all of the jitter: about
        0.6 px on an 80 px eye distance, which is 0.010 of interocular distance. The
        ceiling sits above that, and below the floor a real nose clears.
        """
        return 0.018

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
        self._decided_at = None
        self._pending_level = None
        self._pending_count = 0

    @property
    def verdict(self) -> PresentationVerdict:
        return self._verdict

    @property
    def observation_count(self) -> int:
        return len(self._buffer)

    def observe(self, face: DetectedFace, timestamp: float) -> PresentationVerdict:
        """Record one frame and return the current, stabilised verdict."""
        self._record(face, timestamp)
        if self._decision_due(timestamp):
            self._decided_at = timestamp
            self._publish(self._decide())
        return self._verdict

    def _decision_due(self, timestamp: float) -> bool:
        if self._decided_at is None:
            return True
        return timestamp - self._decided_at >= self.decision_interval

    def _publish(self, candidate: PresentationVerdict) -> None:
        """Publish a level change only after it repeats often enough to be trusted."""
        if candidate.level is self._verdict.level:
            self._verdict = candidate
            self._pending_level = None
            self._pending_count = 0
            return
        if candidate.level is self._pending_level:
            self._pending_count += 1
        else:
            self._pending_level = candidate.level
            self._pending_count = 1
        if self._pending_count < self.stability_windows:
            return
        self._verdict = candidate
        self._pending_level = None
        self._pending_count = 0

    def _record(self, face: DetectedFace, timestamp: float) -> None:
        if face.landmarks is None:
            return
        points = np.asarray(face.landmarks, dtype=np.float32)
        if points.shape != (5, 2):
            return
        scale = interocular_distance(points)
        if scale < _MIN_INTEROCULAR_PX:
            return
        self._buffer.append(FaceObservation(points.copy(), timestamp))
        self._drop_stale(timestamp)

    def _drop_stale(self, timestamp: float) -> None:
        """Forget observations from an earlier look, so a pause cannot be evidence."""
        horizon = timestamp - max(0.5, self._config.observation_window_seconds)
        while self._buffer and self._buffer[0].timestamp < horizon:
            self._buffer.popleft()

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

        # One pair is noisy, the median over the window is not. The residual is already
        # relative to the eye distance and to a plane fitted on the coplanar landmarks.
        nose = median(pair.nose_residual for pair in pairs)
        noise = median(pair.noise_level for pair in pairs)
        motion = median(pair.motion for pair in pairs)
        aggregate = PlanarEvidence(
            nose_residual=nose,
            noise_level=noise,
            motion=motion,
        )
        solid = sum(1 for pair in pairs if self._is_solid(pair))
        planar = sum(1 for pair in pairs if self._is_planar(pair))

        if self._is_solid(aggregate):
            return PresentationVerdict(
                level=SpoofLevel.GENUINE,
                confidence=round(solid / len(pairs), 3),
                detail="The face moved with depth, like a real head.",
                observations=count,
                planar_pairs=planar,
                solid_pairs=solid,
            )
        if self._is_planar(aggregate):
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
            detail="Not enough depth either way yet. Move the head a little more.",
            observations=count,
            planar_pairs=planar,
            solid_pairs=solid,
        )

    def _evidence_pairs(self) -> list[PlanarEvidence]:
        """Compare the earlier half of the window with the later half.

        Pairing across the whole window keeps the sample large, which is what makes
        the median stable, and avoids depending on which frame happens to be newest.
        """
        observations = list(self._buffer)
        half = len(observations) // 2
        if half < 1:
            return []
        step = max(1, half // _MAX_PAIRS)
        evidence: list[PlanarEvidence] = []
        for index in range(0, half, step):
            pair = planar_evidence(
                observations[index].points,
                observations[index + half].points,
                self.min_motion,
            )
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
