"""Turn a raw cosine similarity into a confidence a person can read.

Cosine similarity between two ArcFace embeddings measures distance in a 512-dim
space. It is not a grade, and it is not a probability: for `buffalo_s` the same face
typically lands near 0.65 and a different face near 0.03, so a perfectly correct
match reads "65%" and looks like a coin toss.

The scale below is anchored on the two measured populations instead:

* ``impostor_ceiling``: the highest score seen between *different* people;
* ``genuine_floor``: the lowest score seen between samples of the *same* person.

Those two numbers bracket the only region where a decision is actually uncertain, so
the scale puts them at 5% and 95% and interpolates with a logistic. A score sitting
far above the impostor ceiling then reads as what it is: near-certain, not 65%.

Measured on the shipped `buffalo_s` pack, three identities, five samples each
(`python main.py calibrate` reproduces it on any database):

    same person       n=30   min 0.477   median 0.653   max 0.852
    different people  n=75   min -0.134  median 0.035   max 0.197

Those defaults ship with the app; `calibrate` refits them from your own database once
you have registered everyone, which is the number worth trusting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# The ends of the scale. Never 0 and never 100: a scale that claims certainty would
# hide the fact that every measurement here is statistical.
_LOW = 0.05
_HIGH = 0.95
_CEILING = 0.999
_FLOOR = 0.001
_LOGIT_LOW = math.log(_LOW / (1.0 - _LOW))
_LOGIT_HIGH = math.log(_HIGH / (1.0 - _HIGH))


@dataclass(frozen=True)
class SeparationCalibration:
    """Logistic map from a cosine score to a 1-99 confidence, via two anchors."""

    impostor_ceiling: float
    genuine_floor: float
    genuine_pairs: int = 0
    impostor_pairs: int = 0

    def __post_init__(self) -> None:
        if self.genuine_floor <= self.impostor_ceiling:
            raise ValueError(
                "genuine_floor must sit above impostor_ceiling, got "
                f"{self.genuine_floor} and {self.impostor_ceiling}."
            )

    @property
    def midpoint(self) -> float:
        return (self.impostor_ceiling + self.genuine_floor) / 2.0

    @property
    def slope(self) -> float:
        """Steepness that puts the two anchors at 5% and 95%."""
        return (_LOGIT_HIGH - _LOGIT_LOW) / (self.genuine_floor - self.impostor_ceiling)

    def confidence(self, similarity: float) -> float:
        """Share of the scale, between 0.001 and 0.999, for one cosine score."""
        z = self.slope * (float(similarity) - self.midpoint)
        if z >= 0:
            value = 1.0 / (1.0 + math.exp(-min(z, 60.0)))
        else:
            exponent = math.exp(max(z, -60.0))
            value = exponent / (1.0 + exponent)
        return min(_CEILING, max(_FLOOR, value))

    def percentage(self, similarity: float) -> int:
        """The confidence as a whole percentage, 1 to 99."""
        return max(1, min(99, round(self.confidence(similarity) * 100.0)))

    def display(self, similarity: float) -> str:
        """The confidence with one decimal, which keeps the top of the scale readable.

        Whole numbers would read 99 for every solid match, and a number that never
        moves is indistinguishable from one that is not measured.
        """
        return f"{self.confidence(similarity) * 100.0:.1f}%"

    def describe(self) -> str:
        if self.genuine_pairs and self.impostor_pairs:
            sample = (
                f"fitted on {self.genuine_pairs} same-person and "
                f"{self.impostor_pairs} different-person pairs"
            )
        else:
            sample = "shipped defaults for buffalo_s"
        return (
            f"confidence scale: 5% at cosine {self.impostor_ceiling:.3f}, "
            f"95% at {self.genuine_floor:.3f} ({sample})"
        )


# Shipped defaults: the measurement quoted in the module docstring.
DEFAULT = SeparationCalibration(impostor_ceiling=0.20, genuine_floor=0.48)


def fit(
    genuine: Sequence[float],
    impostor: Sequence[float],
    *,
    percentile: float = 0.98,
) -> SeparationCalibration:
    """Anchor the scale on measured same-person and different-person scores.

    Percentiles rather than the extremes: a single odd pair (a blink, a bad crop)
    should not stretch the scale for everyone.
    """
    if not genuine:
        raise ValueError("Need at least one same-person score to calibrate.")
    if not impostor:
        raise ValueError("Need at least one different-person score to calibrate.")
    ceiling = _percentile(impostor, percentile) + 0.01
    floor = _percentile(genuine, 1.0 - percentile) - 0.01
    if floor <= ceiling:
        # Heavy overlap: fall back to the gap between the medians so the scale still
        # separates something instead of collapsing.
        ceiling = _median(impostor)
        floor = _median(genuine)
    if floor <= ceiling:
        raise ValueError(
            "Same-person and different-person scores overlap too much to calibrate; "
            "record more samples before fitting."
        )
    return SeparationCalibration(
        impostor_ceiling=round(ceiling, 3),
        genuine_floor=round(floor, 3),
        genuine_pairs=len(genuine),
        impostor_pairs=len(impostor),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, fraction)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median(values: Sequence[float]) -> float:
    return _percentile(values, 0.5)
