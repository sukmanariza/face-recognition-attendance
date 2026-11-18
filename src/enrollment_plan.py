"""Staged enrollment: one measurable pose requirement per sample.

Accepting any frame that merely passes the quality gate lets a user move their
head at random and still produce five near-identical embeddings, which is what
makes recognition fragile. Each stage therefore requires a *measured* pose:
front, turned left, turned right, tilted, and finally closer to the camera.

Only metrics that can be computed reliably from the 5-point landmarks are used:
head yaw (horizontal asymmetry of eye-to-nose distances) and roll (eye-line
angle), plus face size for the distance stage. Pitch is not used because a
reliable estimate needs more landmarks than this model provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from config import RegistrationConfig

from .quality import QualityReport


class PoseKind(StrEnum):
    FRONTAL = "frontal"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    TILT = "tilt"
    CLOSER = "closer"


@dataclass(frozen=True)
class PoseStage:
    """One enrollment step and the pose it requires."""

    kind: PoseKind
    title: str
    instruction: str
    yaw_min: float = -1.0
    yaw_max: float = 1.0
    roll_min: float = -90.0
    roll_max: float = 90.0
    min_scale_ratio: float = 0.0


@dataclass(frozen=True)
class StageCheck:
    """Result of comparing a quality report against the active stage."""

    satisfied: bool
    headline: str
    detail: str


def build_plan(config: RegistrationConfig) -> tuple[PoseStage, ...]:
    """Return the stage sequence, truncated to `registration.samples`."""
    front = config.frontal_yaw_tolerance
    turned = config.turned_yaw
    max_roll = config.stage_max_roll_deg
    stages = (
        PoseStage(
            kind=PoseKind.FRONTAL,
            title="Front",
            instruction="Look straight at the camera with a neutral expression.",
            yaw_min=-front,
            yaw_max=front,
            roll_min=-max_roll,
            roll_max=max_roll,
        ),
        PoseStage(
            kind=PoseKind.TURN_LEFT,
            title="Turn left",
            instruction="Slowly turn your head to the left, about 20 degrees.",
            yaw_max=-turned,
            roll_min=-max_roll,
            roll_max=max_roll,
        ),
        PoseStage(
            kind=PoseKind.TURN_RIGHT,
            title="Turn right",
            instruction="Slowly turn your head to the right, about 20 degrees.",
            yaw_min=turned,
            roll_min=-max_roll,
            roll_max=max_roll,
        ),
        PoseStage(
            kind=PoseKind.TILT,
            title="Tilt",
            instruction="Tilt your head slightly to one side, keeping your eyes on the camera.",
            yaw_min=-front * 2,
            yaw_max=front * 2,
            roll_min=config.tilt_roll_min_deg,
            roll_max=config.tilt_roll_max_deg,
        ),
        PoseStage(
            kind=PoseKind.CLOSER,
            title="Closer",
            instruction="Face the camera straight on and move a little closer.",
            yaw_min=-front,
            yaw_max=front,
            roll_min=-max_roll,
            roll_max=max_roll,
            min_scale_ratio=config.closer_scale_ratio,
        ),
    )
    return stages[: max(1, min(config.samples, len(stages)))]


def check_stage(
    stage: PoseStage, report: QualityReport, reference_face_px: int | None
) -> StageCheck:
    """Decide whether `report` satisfies `stage`."""
    yaw = report.yaw_ratio
    roll = report.roll_deg

    if abs(roll) > max(abs(stage.roll_min), abs(stage.roll_max)):
        return StageCheck(
            False,
            "Head tilted too far",
            f"Keep your head upright; tilt stays within {abs(stage.roll_max):.0f} degrees "
            "for this sample.",
        )

    if stage.kind is PoseKind.TILT:
        if abs(roll) < stage.roll_min:
            return StageCheck(
                False,
                "Tilt a little more",
                f"Lean your head to one side by about {stage.roll_min:.0f} degrees.",
            )
    elif not stage.yaw_min <= yaw <= stage.yaw_max:
        if stage.kind is PoseKind.TURN_LEFT:
            return StageCheck(
                False,
                "Turn further to the left",
                "Rotate your head left until the turn reaches the requested angle.",
            )
        if stage.kind is PoseKind.TURN_RIGHT:
            return StageCheck(
                False,
                "Turn further to the right",
                "Rotate your head right until the turn reaches the requested angle.",
            )
        return StageCheck(
            False,
            "Face the camera straight on",
            "Centre your head; this sample needs a frontal view.",
        )

    if stage.min_scale_ratio > 0.0 and reference_face_px:
        required = int(reference_face_px * stage.min_scale_ratio)
        if report.face_size_px < required:
            return StageCheck(
                False,
                "Move closer",
                f"Your face is {report.face_size_px} px; this sample needs at least "
                f"{required} px. Take a small step towards the camera.",
            )

    return StageCheck(True, stage.title, "Pose accepted, hold still.")


def progress_label(stage: PoseStage, index: int, total: int) -> str:
    return f"Sample {index + 1} of {total} · {stage.title}"
