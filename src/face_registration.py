"""Structured face enrollment: staged capture plus persistence.

A sample is accepted only when three independent checks pass, in this order:

1. quality: one face, sharp, well lit, large enough, not turned away
   (`src/quality.py`);
2. pose: the frame matches the requirement of the *current* stage
   (`src/enrollment_plan.py`), so five samples are genuinely different poses
   instead of five frames of the same random movement;
3. consistency: the new sample is close enough to the ones already collected, so
   a different person walking in mid-capture cannot pollute the identity.

The session is written from the inference thread and read from the UI thread, so
every access to the sample list is guarded by one re-entrant lock. ``freeze()``
must be called before samples are persisted: it waits for an in-flight frame to
finish and refuses further samples, so reading never races a write.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from config import RegistrationConfig

from .database import (
    DatabaseError,
    FaceDatabase,
    FaceSample,
)
from .enrollment_plan import PoseStage, build_plan, check_stage
from .face_detector import DetectedFace, FaceDetector
from .liveness import LivenessDetector
from .presentation import PresentationVerdict
from .quality import FaceQualityAssessor, QualityReport
from .similarity import (
    GalleryIndex,
    cosine_similarity_matrix,
    normalize_embedding,
)

logger = logging.getLogger(__name__)


class RegistrationError(RuntimeError):
    """Enrollment cannot proceed or the collected data is unusable."""


class SampleStatus(enum.StrEnum):
    """Outcome of submitting one frame."""

    WAITING = "waiting"
    REJECTED = "rejected"
    CAPTURED = "captured"
    COMPLETE = "complete"


@dataclass(frozen=True)
class SampleAttempt:
    """Result of one `submit_frame` call."""

    status: SampleStatus
    message: str
    collected: int
    required: int
    prompt: str
    quality: QualityReport | None = None
    faces: tuple[DetectedFace, ...] = ()
    stage_title: str = ""
    presentation: PresentationVerdict | None = None

    @property
    def progress(self) -> float:
        if self.required <= 0:
            return 0.0
        return min(1.0, self.collected / self.required)

    @property
    def is_complete(self) -> bool:
        return self.status is SampleStatus.COMPLETE


@dataclass(frozen=True)
class EnrollmentResult:
    """What was written to the database."""

    employee_id: str
    full_name: str
    sample_count: int
    min_consistency: float
    replaced_existing: bool

    def describe(self) -> str:
        action = "updated" if self.replaced_existing else "registered"
        return (
            f"{self.employee_id} ({self.full_name}) {action} with "
            f"{self.sample_count} samples (minimum consistency {self.min_consistency:.3f})."
        )


class FaceRegistrationSession:
    """Collect staged, quality-checked embeddings from camera frames."""

    def __init__(
        self,
        detector: FaceDetector,
        assessor: FaceQualityAssessor,
        config: RegistrationConfig,
        liveness: LivenessDetector | None = None,
        gallery: GalleryIndex | None = None,
    ) -> None:
        self._detector = detector
        self._assessor = assessor
        self._config = config
        self._liveness = liveness
        self._gallery = gallery or GalleryIndex([])
        self._plan = build_plan(config)
        self._samples: list[FaceSample] = []
        self._reference_face_px: int | None = None
        self._last_capture_at: float | None = None
        self._last_faces: tuple[DetectedFace, ...] = ()
        self._duplicate_warning = ""
        self._presentation: PresentationVerdict | None = None
        self._lock = threading.RLock()
        self._frozen = False

    def freeze(self) -> None:
        """Stop accepting samples (idempotent)."""
        with self._lock:
            self._frozen = True

    @property
    def is_frozen(self) -> bool:
        with self._lock:
            return self._frozen

    @property
    def plan(self) -> tuple[PoseStage, ...]:
        return self._plan

    @property
    def collected(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def required(self) -> int:
        return len(self._plan)

    @property
    def is_complete(self) -> bool:
        with self._lock:
            return len(self._samples) >= self.required

    @property
    def current_stage(self) -> PoseStage | None:
        with self._lock:
            index = len(self._samples)
            return self._plan[index] if index < len(self._plan) else None

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._reference_face_px = None
            self._last_capture_at = None
            self._last_faces = ()
            self._duplicate_warning = ""
            self._presentation = None
            self._frozen = False
            if self._liveness is not None:
                self._liveness.reset()

    def submit_frame(self, frame: np.ndarray) -> SampleAttempt:
        """Process one frame; may accept a sample, wait, or reject."""
        with self._lock:
            if self._frozen:
                return self._attempt(SampleStatus.COMPLETE, "This enrollment session is closed.")
            try:
                return self._submit_locked(frame)
            except Exception as exc:
                logger.exception("Enrollment frame failed")
                return self._attempt(SampleStatus.REJECTED, f"Frame could not be analysed: {exc}")

    def _submit_locked(self, frame: np.ndarray) -> SampleAttempt:
        if self.is_complete:
            return self._attempt(SampleStatus.COMPLETE, "All samples are already captured.")

        if not self._capture_due():
            return self._attempt(SampleStatus.WAITING, "Hold still; capturing the next sample.")

        try:
            faces = self._detector.detect(frame, max_faces=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Face detection failed during enrollment: %s", exc)
            self._last_faces = ()
            return self._attempt(SampleStatus.REJECTED, "Detection failed; hold still and retry.")

        self._last_faces = tuple(faces)
        if not faces:
            return self._attempt(SampleStatus.REJECTED, "No face detected.")
        if len(faces) > 1:
            return self._attempt(
                SampleStatus.REJECTED, f"{len(faces)} faces detected; only one person allowed."
            )

        face = faces[0]
        if self._liveness is not None:
            # Observed before the quality gate so the buffer still fills on frames
            # that are rejected for sharpness; the cue needs motion over time.
            liveness = self._liveness.check(frame, face, time.monotonic())
            self._presentation = liveness.verdict
            if liveness.is_live is False:
                return self._attempt(
                    SampleStatus.REJECTED,
                    "This looks like a photo or a screen, not a real face. "
                    "Point the camera at the person directly.",
                )

        report = self._assessor.assess(frame, face)
        if not report.passed:
            return self._attempt(SampleStatus.REJECTED, report.summary(), quality=report)

        stage = self.current_stage
        if stage is None:  # pragma: no cover - guarded by is_complete above
            return self._attempt(SampleStatus.COMPLETE, "All samples are already captured.")
        pose = check_stage(stage, report, self._reference_face_px)
        if not pose.satisfied:
            return self._attempt(SampleStatus.REJECTED, pose.detail, quality=report)

        try:
            self._detector.embed(frame, [face])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding failed during enrollment: %s", exc)
            return self._attempt(SampleStatus.REJECTED, "Embedding failed; hold still and retry.")
        if face.embedding is None:  # pragma: no cover - embed() guarantees this
            return self._attempt(SampleStatus.REJECTED, "No embedding was produced.")

        if self._samples and not self._consistent_with_samples(face.embedding):
            return self._attempt(
                SampleStatus.REJECTED,
                "This does not look like the same person as the earlier samples.",
                quality=report,
            )

        if not self._samples and self._gallery.identity_ids:
            conflict = self._duplicate_identity(face.embedding)
            if conflict is not None:
                self._duplicate_warning = conflict
                return self._attempt(
                    SampleStatus.REJECTED,
                    "This face matches an identity that is already registered.",
                    quality=report,
                )

        if self._reference_face_px is None:
            self._reference_face_px = report.face_size_px

        self._samples.append(
            FaceSample(
                embedding=face.embedding,
                det_score=min(1.0, max(0.0, float(face.det_score))),
                quality_score=report.score,
                yaw_ratio=report.yaw_ratio,
                roll_deg=report.roll_deg,
                model_signature=str(getattr(self._detector, "model_signature", "") or ""),
            )
        )
        self._last_capture_at = time.monotonic()

        if self.is_complete:
            return self._attempt(SampleStatus.COMPLETE, "All samples captured.")
        next_stage = self.current_stage
        return self._attempt(
            SampleStatus.CAPTURED,
            f"Sample saved. Next: {next_stage.title if next_stage else 'finish'}.",
        )

    def _capture_due(self) -> bool:
        if self._last_capture_at is None:
            return True
        return (time.monotonic() - self._last_capture_at) >= self._config.capture_interval_seconds

    def _consistent_with_samples(self, embedding: np.ndarray) -> bool:
        matrix = np.stack([normalize_embedding(sample.embedding) for sample in self._samples])
        scores = cosine_similarity_matrix(embedding, matrix)
        return float(np.max(scores)) >= self._config.min_sample_consistency

    def _duplicate_identity(self, embedding: np.ndarray) -> str | None:
        scores = self._gallery.best_similarity_per_identity(embedding)
        if not scores:
            return None
        employee_id, score = max(scores.items(), key=lambda item: item[1])
        if score < self._config.duplicate_similarity_threshold:
            return None
        name = self._gallery.name_for(employee_id) or employee_id
        return f"{employee_id} - {name} ({score:.2f})"

    def _attempt(
        self,
        status: SampleStatus,
        message: str,
        quality: QualityReport | None = None,
    ) -> SampleAttempt:
        stage = self.current_stage
        return SampleAttempt(
            status=status,
            message=message,
            collected=self.collected,
            required=self.required,
            prompt=stage.instruction if stage else "All samples captured.",
            quality=quality,
            faces=self._last_faces,
            stage_title=stage.title if stage else "Done",
            presentation=self._presentation,
        )

    def min_consistency(self) -> float:
        """Lowest similarity of any sample against the mean of all samples."""
        with self._lock:
            return self._min_consistency_locked()

    def _min_consistency_locked(self) -> float:
        if len(self._samples) < 2:
            return 1.0
        matrix = np.stack([normalize_embedding(sample.embedding) for sample in self._samples])
        mean = normalize_embedding(matrix.mean(axis=0))
        return float(np.min(cosine_similarity_matrix(mean, matrix)))

    def build_samples(self) -> tuple[FaceSample, ...]:
        """Validate the whole set and return the samples."""
        with self._lock:
            if len(self._samples) < self.required:
                raise RegistrationError(
                    f"Capture incomplete: {len(self._samples)} of {self.required} samples."
                )
            consistency = self._min_consistency_locked()
            if consistency < self._config.min_sample_consistency:
                raise RegistrationError(
                    f"Samples are inconsistent (lowest similarity {consistency:.3f} < "
                    f"{self._config.min_sample_consistency:.3f}). Record again with better "
                    "lighting and follow the pose instructions."
                )
            if self._duplicate_warning:
                raise RegistrationError(
                    f"This face matches an identity that is already registered "
                    f"({self._duplicate_warning})."
                )
            return tuple(self._samples)


class FaceRegistrar:
    """Write enrollment results to the database with duplicate protection."""

    def __init__(self, database: FaceDatabase, config: RegistrationConfig) -> None:
        self._database = database
        self._config = config

    def save_new(
        self,
        employee_id: str,
        full_name: str,
        samples: Sequence[FaceSample],
        min_consistency: float,
    ) -> EnrollmentResult:
        self._reject_duplicate_face(samples, exclude_id=None)
        record = self._database.add_employee(employee_id, full_name, list(samples))
        return EnrollmentResult(
            record.employee_id, record.full_name, record.sample_count, min_consistency, False
        )

    def save_update(
        self,
        employee_id: str,
        samples: Sequence[FaceSample],
        min_consistency: float,
        full_name: str | None = None,
    ) -> EnrollmentResult:
        self._reject_duplicate_face(samples, exclude_id=employee_id)
        record = self._database.replace_samples(employee_id, list(samples))
        if full_name and full_name != record.full_name:
            record = self._database.rename_employee(employee_id, full_name)
        return EnrollmentResult(
            record.employee_id, record.full_name, record.sample_count, min_consistency, True
        )

    def _reject_duplicate_face(
        self, samples: Sequence[FaceSample], exclude_id: str | None
    ) -> None:
        """Refuse to register a face that already belongs to another identity."""
        try:
            records = [
                record
                for record in self._database.load_gallery()
                if exclude_id is None or record.employee_id != exclude_id
            ]
        except DatabaseError:
            return
        if not records:
            return

        gallery = GalleryIndex(records)
        best_id, best_score = None, -1.0
        for sample in samples:
            scores = gallery.best_similarity_per_identity(sample.embedding)
            if not scores:
                continue
            candidate, score = max(scores.items(), key=lambda item: item[1])
            if score > best_score:
                best_id, best_score = candidate, score

        if best_id is not None and best_score >= self._config.duplicate_similarity_threshold:
            name = gallery.name_for(best_id) or best_id
            raise RegistrationError(
                f"This face already belongs to {best_id} - {name} (similarity {best_score:.2f}). "
                "Use 'Re-record face' for that identity instead of creating a new one."
            )
