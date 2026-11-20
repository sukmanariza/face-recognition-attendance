"""Turn camera conditions into one actionable instruction per frame.

A rejected sample is not a user error; it is one measurable condition that is not
met. This module picks the condition that blocks capture most and says how to fix
it, so the screen never just says "rejected".

Ordering follows impact on the embedding: lighting and occlusion come first,
because a dark or obstructed face yields a misleading embedding even when the pose
is correct. Pose requirements from `src/enrollment_plan.py` come last, since they
only matter once the image itself is usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from config import QualityConfig

from .enrollment_plan import PoseStage, check_stage
from .face_registration import SampleAttempt, SampleStatus
from .quality import QualityReport


class GuidanceLevel(StrEnum):
    """Severity, which drives colour and tone in the interface."""

    READY = "ready"
    ADJUST = "adjust"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Guidance:
    level: GuidanceLevel
    headline: str
    detail: str

    @property
    def is_ready(self) -> bool:
        return self.level is GuidanceLevel.READY


def assess_attempt(
    attempt: SampleAttempt,
    stage: PoseStage | None,
    config: QualityConfig,
    reference_face_px: int | None = None,
) -> Guidance:
    """Instruction for one enrollment frame."""
    if attempt.status is SampleStatus.COMPLETE:
        return Guidance(GuidanceLevel.READY, "All samples captured", "Saving to the database...")
    if attempt.status is SampleStatus.CAPTURED:
        return Guidance(
            GuidanceLevel.READY, "Sample saved", stage.instruction if stage else "Next sample."
        )
    if attempt.status is SampleStatus.WAITING:
        return Guidance(
            GuidanceLevel.ADJUST, "Hold still", "The camera is taking the next sample."
        )

    if not attempt.faces:
        return Guidance(
            GuidanceLevel.BLOCKED,
            "No face detected",
            "Place your face inside the frame and look towards the camera.",
        )
    if len(attempt.faces) > 1:
        return Guidance(
            GuidanceLevel.BLOCKED,
            f"{len(attempt.faces)} faces detected",
            "Only one person should be in front of the camera.",
        )
    if attempt.quality is None:
        return Guidance(GuidanceLevel.ADJUST, "Adjusting", attempt.message)

    quality = assess_quality(attempt.quality, config)
    if not quality.is_ready:
        return quality

    if stage is not None:
        check = check_stage(stage, attempt.quality, reference_face_px)
        if not check.satisfied:
            return Guidance(GuidanceLevel.ADJUST, check.headline, check.detail)
        return Guidance(GuidanceLevel.READY, "Pose accepted", "Hold still to capture.")

    return Guidance(GuidanceLevel.READY, "Ready", "Hold still until the sample is saved.")


def assess_quality(report: QualityReport, config: QualityConfig) -> Guidance:
    """Instruction for the measurable image-quality conditions alone."""
    if report.passed:
        return Guidance(GuidanceLevel.READY, "Image is good", "Hold still.")

    if report.brightness < config.brightness_min:
        return Guidance(
            GuidanceLevel.BLOCKED,
            "Too dark",
            f"Brightness is {report.brightness:.0f}; at least {config.brightness_min:.0f} is "
            "needed. Add light in front of your face, not behind it.",
        )
    if report.brightness > config.brightness_max:
        return Guidance(
            GuidanceLevel.BLOCKED,
            "Too bright",
            f"Brightness is {report.brightness:.0f}; the limit is {config.brightness_max:.0f}. "
            "Move away from a lamp or window shining directly on your face.",
        )
    if report.contrast < config.contrast_min:
        return Guidance(
            GuidanceLevel.BLOCKED,
            "Backlit",
            "Your face is darker than the background. Move so the light comes from in front.",
        )
    if report.blur_score < config.blur_threshold:
        return Guidance(
            GuidanceLevel.ADJUST,
            "Image is blurry",
            "Hold still for a moment, and clean the lens if it looks smudged.",
        )
    if report.face_size_px < config.min_face_size_px:
        return Guidance(
            GuidanceLevel.ADJUST,
            "Too far from the camera",
            f"Your face is {report.face_size_px} px; at least {config.min_face_size_px} px is "
            "needed. Move closer.",
        )
    if abs(report.yaw_ratio) > config.max_yaw_ratio:
        return Guidance(
            GuidanceLevel.ADJUST, "Head turned too far", "Keep your face towards the camera."
        )
    if abs(report.roll_deg) > config.max_roll_deg:
        return Guidance(
            GuidanceLevel.ADJUST,
            "Head tilted too far",
            f"Tilt is {abs(report.roll_deg):.0f} degrees; the limit is "
            f"{config.max_roll_deg:.0f}. Straighten your head.",
        )
    if report.det_score < config.min_det_score:
        return Guidance(
            GuidanceLevel.ADJUST,
            "Face partly hidden",
            "Make sure nothing covers your face: hands, mask, or hair.",
        )

    return Guidance(GuidanceLevel.ADJUST, "Sample not usable yet", report.summary())


def checklist_hints(config: QualityConfig) -> tuple[str, ...]:
    """Setup advice that the model cannot measure.

    This model has no attribute head, so glasses and makeup cannot be detected.
    These items are shown for the user to confirm, never as detected facts.
    """
    return (
        (
            "Remove glasses if you do not always wear them; lens reflections change "
            "the eye region of the embedding."
        ),
        "Face a light source such as a desk lamp or a window in front of you.",
        (
            f"Sit about 40-60 cm from the camera so your face is at least "
            f"{config.min_face_size_px} px wide."
        ),
        "Move hair away from your forehead and cheeks, and do not wear a mask.",
    )
