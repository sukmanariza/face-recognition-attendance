"""Presentation check wiring: one detector behind a small, stable contract.

The measurement itself lives in `src/presentation.py`. This module gives other code
one way to ask "is this a real face in front of the camera?" and keeps the answer
tri-state, because a single frame cannot tell. Three callers use it:

* the recognition session labels each face;
* the enrollment session refuses to store a face that moved like a photo;
* the CLI preview prints the same label for headless machines.

`mode` decides how much the answer is allowed to do:

* ``off``: nothing is measured, and every answer is "unknown";
* ``advisory``: the label is shown, but a match is still reported;
* ``enforce``: a detected photo or screen is not accepted as an identity.

A detected attack is refused in the enrollment path in every mode except ``off``,
because a spoofed sample would poison the gallery for as long as that identity
exists. "Unknown" is never treated as an attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from config import LivenessConfig

from .face_detector import DetectedFace
from .presentation import PresentationGuard, PresentationVerdict, SpoofLevel

__all__ = [
    "DisabledLivenessDetector",
    "LivenessDetector",
    "LivenessResult",
    "PresentationLivenessDetector",
    "SpoofLevel",
    "available_detectors",
    "build_liveness_detector",
]


@dataclass(frozen=True)
class LivenessResult:
    """Outcome of one presentation check."""

    is_live: bool | None
    score: float | None
    reason: str
    verdict: PresentationVerdict | None = None

    @property
    def label(self) -> str:
        return self.verdict.label if self.verdict is not None else "Not checked"


@runtime_checkable
class LivenessDetector(Protocol):
    """Contract for a presentation-attack detector."""

    name: str

    def check(
        self, frame: np.ndarray, face: DetectedFace, timestamp: float = 0.0
    ) -> LivenessResult:
        """Judge whether the face in ``frame`` comes from a live person."""
        ...

    def reset(self) -> None:
        """Forget everything collected so far."""
        ...


class DisabledLivenessDetector:
    """Stand-in used when the check is off.

    Always answers "unknown" rather than "live", so nothing downstream can mistake
    an unchecked face for a verified one.
    """

    name = "disabled"

    def check(
        self, frame: np.ndarray, face: DetectedFace, timestamp: float = 0.0
    ) -> LivenessResult:
        return LivenessResult(
            is_live=None,
            score=None,
            reason="Presentation check is off in Settings.",
        )

    def reset(self) -> None:
        return None


class PresentationLivenessDetector:
    """Adapts `PresentationGuard` to the detector contract."""

    name = "planar-parallax"

    def __init__(self, config: LivenessConfig) -> None:
        self._guard = PresentationGuard(config)

    @property
    def guard(self) -> PresentationGuard:
        return self._guard

    def check(
        self, frame: np.ndarray, face: DetectedFace, timestamp: float = 0.0
    ) -> LivenessResult:
        verdict = self._guard.observe(face, timestamp)
        return LivenessResult(
            is_live=verdict.is_live,
            score=verdict.confidence,
            reason=verdict.detail,
            verdict=verdict,
        )

    def reset(self) -> None:
        self._guard.reset()


def available_detectors() -> tuple[str, ...]:
    """Names of the presentation detectors this build ships."""
    return ("planar-parallax",)


def build_liveness_detector(config: LivenessConfig) -> LivenessDetector:
    """Return the detector for the configured mode."""
    if not config.enabled:
        return DisabledLivenessDetector()
    return PresentationLivenessDetector(config)
