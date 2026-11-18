"""Pick inference settings that fit the machine running the app.

Four tiers cover the realistic range, from a 1 GB single-board computer up to a
workstation. `auto` derives the tier from CPU cores, RAM and the board itself, so a
Raspberry Pi gets the lightest configuration without the user tuning anything.

The frame rate is not capped by the app: the camera is asked for what it can do, and
the display paints whenever a new result exists. The tier only decides how often the
expensive stages run, which is what keeps a slow board responsive instead of
queueing work it cannot finish.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum

from config import AppConfig

from .hardware import Hardware


class Tier(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"


@dataclass(frozen=True)
class TierProfile:
    """Inference and capture settings for one tier."""

    label: str
    det_size: tuple[int, int]
    intra_op_threads: int
    inter_op_threads: int
    detection_interval: int
    embedding_interval_seconds: float
    requested_fps: int


PROFILES: dict[Tier, TierProfile] = {
    Tier.MINIMAL: TierProfile(
        label="Minimal (1 GB board, one or two slow cores)",
        det_size=(256, 256),
        intra_op_threads=1,
        inter_op_threads=1,
        detection_interval=5,
        embedding_interval_seconds=1.5,
        requested_fps=15,
    ),
    Tier.LOW: TierProfile(
        label="Low power (single-board computer, 2 to 4 cores)",
        det_size=(320, 320),
        intra_op_threads=1,
        inter_op_threads=1,
        detection_interval=3,
        embedding_interval_seconds=1.0,
        requested_fps=30,
    ),
    Tier.BALANCED: TierProfile(
        label="Balanced (4-core laptop)",
        det_size=(320, 320),
        intra_op_threads=2,
        inter_op_threads=1,
        detection_interval=2,
        embedding_interval_seconds=0.5,
        requested_fps=60,
    ),
    Tier.HIGH: TierProfile(
        label="High (6+ cores)",
        det_size=(480, 480),
        intra_op_threads=4,
        inter_op_threads=1,
        detection_interval=1,
        embedding_interval_seconds=0.4,
        requested_fps=60,
    ),
}

_LOW_RAM_MB = 2048
_MINIMAL_RAM_MB = 1024
_BALANCED_RAM_MB = 6144
_LOW_CORES = 2
_MINIMAL_CORES = 1
_BALANCED_CORES = 4


def choose_tier(hardware: Hardware, requested: str) -> Tier:
    """Resolve `auto` from the hardware, otherwise honour the explicit choice."""
    if requested != "auto":
        return Tier(requested)
    if hardware.cpu_cores <= _MINIMAL_CORES:
        return Tier.MINIMAL
    if hardware.total_ram_mb and hardware.total_ram_mb <= _MINIMAL_RAM_MB:
        return Tier.MINIMAL
    if hardware.is_single_board and hardware.cpu_cores <= _LOW_CORES:
        return Tier.MINIMAL
    if hardware.is_single_board:
        return Tier.LOW
    if hardware.cpu_cores <= _LOW_CORES:
        return Tier.LOW
    if hardware.total_ram_mb and hardware.total_ram_mb <= _LOW_RAM_MB:
        return Tier.LOW
    if hardware.cpu_cores <= _BALANCED_CORES:
        return Tier.BALANCED
    if hardware.total_ram_mb and hardware.total_ram_mb <= _BALANCED_RAM_MB:
        return Tier.BALANCED
    return Tier.HIGH


def apply_profile(config: AppConfig, hardware: Hardware) -> tuple[AppConfig, Tier]:
    """Return a config tuned for `hardware`, plus the tier that was applied.

    Only performance-related fields change; accuracy settings and paths are left
    untouched so a tier switch never silently alters recognition behaviour.
    """
    tier = choose_tier(hardware, config.performance.profile)
    profile = PROFILES[tier]
    tuned = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            det_size=profile.det_size,
            intra_op_threads=profile.intra_op_threads,
            inter_op_threads=profile.inter_op_threads,
        ),
        recognition=dataclasses.replace(
            config.recognition,
            detection_interval=profile.detection_interval,
            embedding_interval_seconds=profile.embedding_interval_seconds,
        ),
        camera=dataclasses.replace(config.camera, requested_fps=profile.requested_fps),
    )
    return tuned, tier


def describe_profile(tier: Tier, hardware: Hardware) -> str:
    profile = PROFILES[tier]
    width, height = profile.det_size
    return (
        f"{profile.label} · detector {width}×{height} · "
        f"{profile.intra_op_threads} inference thread(s) · {hardware.summary}"
    )
