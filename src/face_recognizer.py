"""Per-frame recognition: detection, embedding, matching, and short tracking.

Three budgets keep this cheap on a CPU:

* ``detection_interval``: detect on every Nth frame and reuse the last boxes in
  between, so box edges trail the face slightly while the cost stays predictable;
* ``embedding_interval_seconds``: embed each tracked face at most that often;
* ``confirmation_hits``: a name is shown only after the same identity wins that
  many consecutive checks, which removes one-frame flips.

``smoothing_alpha`` blends similarity over time so the displayed percentage does
not jitter. Matching itself always uses the raw embedding: smoothing the score is
for display and confirmation, never for the accept/reject decision.
"""

from __future__ import annotations

import enum
import itertools
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from config import RecognitionConfig

from .attendance import AttendanceDecision, AttendanceService
from .calibration import DEFAULT as DEFAULT_CALIBRATION
from .calibration import SeparationCalibration
from .face_detector import DetectedFace, FaceDetector
from .liveness import LivenessDetector
from .presentation import PresentationVerdict
from .similarity import (
    FaceMatcher,
    GalleryIndex,
    MatchResult,
    to_percentage,
)

logger = logging.getLogger(__name__)


class FaceState(enum.StrEnum):
    """What the UI should show for one face."""

    MATCHED = "matched"
    VERIFYING = "verifying"
    UNKNOWN = "unknown"


@dataclass
class RecognizedFace:
    """One face in the current frame with its identity decision."""

    face: DetectedFace
    track_id: int
    state: FaceState
    similarity: float
    similarity_smoothed: float
    hits: int
    employee_id: str | None = None
    full_name: str | None = None
    reason: str = ""
    liveness: bool | None = None
    presentation: PresentationVerdict | None = None
    attendance: AttendanceDecision | None = None
    embedding: np.ndarray | None = None
    calibration: SeparationCalibration = DEFAULT_CALIBRATION

    @property
    def bbox(self) -> np.ndarray:
        return self.face.bbox

    @property
    def landmarks(self) -> np.ndarray | None:
        return self.face.landmarks

    @property
    def det_score(self) -> float:
        return self.face.det_score

    @property
    def accepted(self) -> bool:
        return self.state is FaceState.MATCHED

    @property
    def percentage(self) -> int:
        """The raw cosine score as a whole percentage.

        This is the same scale as `recognition.similarity_threshold`, so it is what
        the threshold is compared against. It is not a confidence; `confidence` is.
        """
        return to_percentage(self.similarity_smoothed)

    @property
    def confidence(self) -> int:
        """Calibrated confidence, 1 to 99, that this is the same person."""
        return self.calibration.percentage(self.similarity_smoothed)

    @property
    def confidence_text(self) -> str:
        return self.calibration.display(self.similarity_smoothed)

    @property
    def display_name(self) -> str:
        if self.accepted:
            return self.full_name or "Unknown"
        if self.state is FaceState.VERIFYING:
            return "Verifying"
        return "Unknown"

    @property
    def display_id(self) -> str:
        if self.accepted:
            return self.employee_id or "UNKNOWN"
        return "VERIFYING" if self.state is FaceState.VERIFYING else "UNKNOWN"

    @property
    def label(self) -> str:
        """Single line for the on-frame HUD."""
        if self.presentation is not None:
            return f"{self.display_name}  {self.confidence_text}  [{self.presentation.short_label}]"
        return f"{self.display_name}  {self.confidence_text}"


@dataclass(frozen=True)
class FrameRecognition:
    """Everything the UI needs about one processed frame."""

    faces: tuple[RecognizedFace, ...] = ()
    inference_ms: float = 0.0
    fps: float = 0.0
    threshold: float = 0.0
    timestamp: float = 0.0

    @property
    def matched(self) -> tuple[RecognizedFace, ...]:
        return tuple(face for face in self.faces if face.accepted)

    @property
    def unknown(self) -> tuple[RecognizedFace, ...]:
        return tuple(face for face in self.faces if not face.accepted)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def summary(self) -> str:
        names = ", ".join(face.display_name for face in self.matched) or "no match"
        return f"{self.face_count} face(s), {names}, {self.inference_ms:.1f} ms"


@dataclass
class _Track:
    """Short-lived track that carries the last embedding and hit count."""

    track_id: int
    bbox: np.ndarray
    landmarks: np.ndarray | None
    det_score: float
    last_seen: float
    embedding: np.ndarray | None = None
    match: MatchResult | None = None
    similarity_smoothed: float = 0.0
    hits: int = 0
    liveness: bool | None = None
    presentation: PresentationVerdict | None = None
    attendance: AttendanceDecision | None = None
    last_embed_at: float = field(default=0.0)


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


class FaceRecognitionSession:
    """Stateful recognizer: one instance per camera stream, one thread."""

    def __init__(
        self,
        detector: FaceDetector,
        gallery: GalleryIndex | None,
        config: RecognitionConfig,
        liveness: LivenessDetector | None = None,
        enforce_presentation: bool = False,
        attendance: AttendanceService | None = None,
    ) -> None:
        self._detector = detector
        self._gallery = gallery or GalleryIndex([])
        self._config = config
        self._liveness = liveness
        self._enforce_presentation = enforce_presentation
        self._attendance = attendance
        self._matcher = FaceMatcher(config.similarity_threshold, config.ambiguity_margin)
        self._calibration = config.calibration
        self._tracks: dict[int, _Track] = {}
        self._track_ids = itertools.count(1)
        self._frame_index = 0
        self._last_detections: list[DetectedFace] = []
        self._fps = 0.0
        self._last_frame_at: float | None = None
        self._presentation_track: int | None = None

    @property
    def gallery(self) -> GalleryIndex:
        return self._gallery

    def set_gallery(self, gallery: GalleryIndex) -> None:
        """Swap the gallery (after enrollment) without losing track state."""
        self._gallery = gallery
        self.reset()

    @property
    def threshold(self) -> float:
        return self._matcher.threshold

    @property
    def calibration(self) -> SeparationCalibration:
        return self._calibration

    @property
    def config(self) -> RecognitionConfig:
        return self._config

    def update_config(self, config: RecognitionConfig) -> None:
        """Apply changed thresholds without rebuilding the session."""
        self._config = config
        self._matcher = FaceMatcher(config.similarity_threshold, config.ambiguity_margin)
        self._calibration = config.calibration

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._last_detections.clear()
        self._frame_index = 0
        self._fps = 0.0
        self._last_frame_at = None
        self._presentation_track = None
        if self._liveness is not None:
            self._liveness.reset()

    def process(self, image: np.ndarray, timestamp: float | None = None) -> FrameRecognition:
        """Detect, embed (throttled), match, and confirm faces in one frame."""
        started = time.perf_counter()
        now = timestamp if timestamp is not None else time.monotonic()
        self._frame_index += 1

        if self._last_frame_at is not None:
            delta = now - self._last_frame_at
            if delta > 1e-6:
                instant = 1.0 / delta
                # Exponential average: stable readout without a ring buffer.
                self._fps = instant if self._fps <= 0.0 else 0.85 * self._fps + 0.15 * instant
        self._last_frame_at = now

        detections = self._detect(image)
        tracks = self._assign_tracks(detections, now)
        self._prune_tracks(now)

        faces: list[RecognizedFace] = []
        for track in tracks:
            self._observe_presentation(image, track, now, single=len(tracks) == 1)
            self._refresh_track(image, track, now)
            faces.append(self._to_recognized(track, now))

        faces.sort(key=lambda item: float(item.face.area), reverse=True)
        inference_ms = (time.perf_counter() - started) * 1000.0
        return FrameRecognition(
            faces=tuple(faces),
            inference_ms=inference_ms,
            fps=self._fps,
            threshold=self._matcher.threshold,
            timestamp=now,
        )

    def _detect(self, image: np.ndarray) -> list[DetectedFace]:
        interval = max(1, self._config.detection_interval)
        # interval = 1 detects on every frame; interval = 3 on frames 3, 6, 9, ...
        if self._frame_index % interval == 0 or not self._last_detections:
            self._last_detections = self._detector.detect(image, max_faces=self._config.max_faces)
        return self._last_detections

    def _assign_tracks(self, detections: list[DetectedFace], now: float) -> list[_Track]:
        """Pair each detection with the track it overlaps most (greedy, by IoU)."""
        unmatched_tracks = list(self._tracks.values())
        assigned: list[_Track] = []

        for detection in detections:
            best: _Track | None = None
            best_iou = self._config.iou_threshold
            for track in unmatched_tracks:
                score = _iou(track.bbox, detection.bbox)
                if score >= best_iou:
                    best, best_iou = track, score

            if best is None:
                track = _Track(
                    track_id=next(self._track_ids),
                    bbox=detection.bbox,
                    landmarks=detection.landmarks,
                    det_score=detection.det_score,
                    last_seen=now,
                )
                self._tracks[track.track_id] = track
            else:
                unmatched_tracks.remove(best)
                best.bbox = detection.bbox
                best.landmarks = detection.landmarks
                best.det_score = detection.det_score
                best.last_seen = now
                track = best
            assigned.append(track)

        return assigned

    def _prune_tracks(self, now: float) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.last_seen > self._config.track_max_age_seconds
        ]
        for track_id in stale:
            self._tracks.pop(track_id, None)

    def _observe_presentation(
        self, image: np.ndarray, track: _Track, now: float, single: bool
    ) -> None:
        """Feed one frame to the presentation check.

        The check compares one face against itself over time, so it only runs while
        a single face is in view; with two people the samples would be mixed.
        """
        if self._liveness is None or track.landmarks is None:
            return
        if not single:
            # A second person would put their landmarks into the same series.
            if self._presentation_track is not None:
                self._liveness.reset()
                self._presentation_track = None
            track.presentation = None
            return
        if self._presentation_track != track.track_id:
            # A different person: the previous series says nothing about this one.
            self._liveness.reset()
            self._presentation_track = track.track_id
        face = DetectedFace(
            bbox=track.bbox, det_score=track.det_score, landmarks=track.landmarks
        )
        result = self._liveness.check(image, face, now)
        track.presentation = result.verdict
        track.liveness = result.is_live

    def _refresh_track(self, image: np.ndarray, track: _Track, now: float) -> None:
        """Embed and match a track when its embedding interval has elapsed."""
        due = (
            track.embedding is None
            or (now - track.last_embed_at) >= self._config.embedding_interval_seconds
        )
        if not due:
            return

        face = DetectedFace(
            bbox=track.bbox,
            det_score=track.det_score,
            landmarks=track.landmarks,
        )
        try:
            self._detector.embed(image, [face])
        except Exception as exc:  # noqa: BLE001 - a bad frame must not stop the stream
            logger.debug("Embedding skipped: %s", exc)
            return
        if face.embedding is None:  # pragma: no cover - embed() either fills or raises
            return

        track.embedding = face.embedding
        track.last_embed_at = now
        track.match = self._matcher.match(face.embedding, self._gallery)

        alpha = self._config.smoothing_alpha
        target = float(track.match.similarity)
        if track.hits == 0:
            track.similarity_smoothed = target
        else:
            track.similarity_smoothed = alpha * target + (1.0 - alpha) * track.similarity_smoothed

        confirmed = track.match.accepted and not self._presentation_blocks(track)
        if confirmed:
            track.hits += 1
        else:
            track.hits = 0

        if confirmed and track.hits >= self._config.confirmation_hits:
            self._record_attendance(track)

    def _record_attendance(self, track: _Track) -> None:
        """Mark presence once a match is confirmed.

        Called from the embedding path only, so the cost is one call per
        ``embedding_interval_seconds`` at most, and the service itself writes at most
        once per ``attendance.refresh_seconds``.
        """
        service = self._attendance
        match = track.match
        if service is None or match is None or not match.employee_id:
            return
        decision = service.record(match.employee_id, track.presentation)
        if decision.recorded:
            track.attendance = decision

    def _presentation_blocks(self, track: _Track) -> bool:
        """Only the enforcing mode may stop a name from appearing."""
        presentation = track.presentation
        return (
            self._enforce_presentation
            and presentation is not None
            and presentation.is_suspected
        )

    def _to_recognized(self, track: _Track, now: float) -> RecognizedFace:
        face = DetectedFace(
            bbox=track.bbox,
            det_score=track.det_score,
            landmarks=track.landmarks,
            embedding=track.embedding,
        )
        if track.match is None:
            return RecognizedFace(
                face=face,
                track_id=track.track_id,
                state=FaceState.UNKNOWN,
                similarity=0.0,
                similarity_smoothed=0.0,
                hits=0,
                reason="analysing",
                calibration=self._calibration,
                presentation=track.presentation,
                attendance=track.attendance,
                embedding=track.embedding,
            )

        matcher_result = track.match
        presentation = track.presentation
        if self._presentation_blocks(track) and presentation is not None:
            state = FaceState.UNKNOWN
            reason = f"{presentation.label}: {presentation.detail}"
        elif not matcher_result.accepted:
            state = FaceState.UNKNOWN
            reason = matcher_result.reason
        elif track.hits < self._config.confirmation_hits:
            state = FaceState.VERIFYING
            reason = f"confirming {track.hits}/{self._config.confirmation_hits}"
        else:
            state = FaceState.MATCHED
            reason = matcher_result.reason

        return RecognizedFace(
            face=face,
            track_id=track.track_id,
            state=state,
            similarity=matcher_result.similarity,
            similarity_smoothed=track.similarity_smoothed,
            hits=track.hits,
            employee_id=matcher_result.employee_id,
            full_name=matcher_result.full_name,
            reason=reason,
            liveness=track.liveness,
            presentation=presentation,
            attendance=track.attendance,
            embedding=track.embedding,
            calibration=self._calibration,
        )
