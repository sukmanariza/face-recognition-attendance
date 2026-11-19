"""Measure how well this database separates people, so the scale is not a guess.

`python main.py calibrate` calls this. It compares every stored sample with every
other one and splits the resulting cosine scores into two populations:

* **same person**: two samples of one identity, which is what a correct match looks
  like;
* **different people**: samples of two identities, which is what a mistake looks like.

Where those two populations stop overlapping is where a threshold belongs, and the
same two numbers anchor the confidence scale.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .similarity import GalleryIndex


@dataclass(frozen=True)
class SeparationMeasurement:
    """Cosine scores of same-person and different-person pairs, plus their spread."""

    genuine: tuple[float, ...]
    impostor: tuple[float, ...]

    @property
    def genuine_median(self) -> float:
        return _median(self.genuine)

    @property
    def impostor_median(self) -> float:
        return _median(self.impostor)

    @property
    def overlap(self) -> bool:
        """True when some different-person pair scores above a same-person pair."""
        return bool(self.genuine) and bool(self.impostor) and max(self.impostor) >= min(
            self.genuine
        )

    def describe(self) -> str:
        lines = [
            f"same person      n={len(self.genuine):3d}  "
            f"min {min(self.genuine):.3f}  median {self.genuine_median:.3f}  "
            f"max {max(self.genuine):.3f}",
            f"different people n={len(self.impostor):3d}  "
            f"min {min(self.impostor):.3f}  median {self.impostor_median:.3f}  "
            f"max {max(self.impostor):.3f}",
        ]
        if self.overlap:
            lines.append("the two populations overlap; add more samples before trusting a fit")
        return "\n".join(lines)

    def threshold_advice(self, threshold: float) -> str:
        """Say whether the configured threshold sits in the gap or inside a population."""
        below = sum(1 for value in self.genuine if value < threshold)
        share = below / len(self.genuine) if self.genuine else 0.0
        if below:
            return (
                f"threshold {threshold:.2f} rejects {below} of {len(self.genuine)} "
                f"same-person pairs ({share:.0%}); the gap between the populations is "
                f"{max(self.impostor):.3f} to {min(self.genuine):.3f}, so a threshold in "
                "that gap accepts everyone and still excludes every impostor."
            )
        return (
            f"threshold {threshold:.2f} accepts every same-person pair in this database "
            f"and sits above the closest impostor ({max(self.impostor):.3f})."
        )


def measure(gallery: GalleryIndex) -> SeparationMeasurement | None:
    """Score every pair in the gallery, or return None when there is too little data."""
    identities = gallery.identity_ids
    stored = {identity: gallery.matches_for(identity) for identity in identities}
    usable = {
        identity: np.asarray(samples)
        for identity, samples in stored.items()
        if samples is not None and len(samples) >= 2
    }

    genuine: list[float] = []
    for matrix in usable.values():
        for first, second in itertools.combinations(range(len(matrix)), 2):
            genuine.append(_cosine(matrix[first], matrix[second]))

    impostor: list[float] = []
    for left, right in itertools.combinations(sorted(usable), 2):
        for row_left in usable[left]:
            for row_right in usable[right]:
                impostor.append(_cosine(row_left, row_right))

    if not genuine or not impostor:
        return None
    return SeparationMeasurement(genuine=tuple(genuine), impostor=tuple(impostor))


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = first / np.linalg.norm(first)
    right = second / np.linalg.norm(second)
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
