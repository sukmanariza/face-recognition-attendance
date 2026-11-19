"""The confidence scale: turning a cosine score into a number a person can read.

The point of these tests is that the scale never flatters. A score at the impostor
ceiling must read low, a score at the genuine floor must read high, and the whole
thing must stay monotone and bounded so the UI cannot show 0% or 100%.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.calibration import DEFAULT, SeparationCalibration, fit
from src.calibration_audit import SeparationMeasurement, measure
from src.similarity import GalleryIndex, IdentityRecord


def identity(employee_id: str, seed: int, samples: int = 3) -> IdentityRecord:
    generator = np.random.default_rng(seed)
    base = generator.normal(size=512).astype(np.float32)
    base /= np.linalg.norm(base)
    rows = []
    for _ in range(samples):
        noise = generator.normal(scale=0.05, size=512).astype(np.float32)
        rows.append(base + noise)
    return IdentityRecord(
        employee_id=employee_id, full_name=f"Person {employee_id}", embeddings=np.stack(rows)
    )


class ScaleTests(unittest.TestCase):
    def test_the_anchor_points_land_on_the_expected_ends(self) -> None:
        calibration = SeparationCalibration(impostor_ceiling=0.20, genuine_floor=0.48)

        self.assertAlmostEqual(calibration.confidence(0.20), 0.05, places=6)
        self.assertAlmostEqual(calibration.confidence(0.48), 0.95, places=6)

    def test_a_typical_correct_match_reads_high(self) -> None:
        # 0.65 is the measured median for the same person with buffalo_s.
        self.assertGreaterEqual(DEFAULT.percentage(0.65), 90)

    def test_a_typical_stranger_reads_low(self) -> None:
        # 0.03 is the measured median for two different people with buffalo_s.
        self.assertLessEqual(DEFAULT.percentage(0.03), 5)

    def test_the_scale_is_monotone(self) -> None:
        scores = [index / 100 for index in range(-20, 101)]
        values = [DEFAULT.confidence(score) for score in scores]

        self.assertEqual(values, sorted(values))

    def test_it_never_claims_certainty_or_impossibility(self) -> None:
        self.assertGreaterEqual(DEFAULT.confidence(-1.0), 0.001)
        self.assertLessEqual(DEFAULT.confidence(1.0), 0.999)
        self.assertLess(DEFAULT.percentage(1.0), 100)
        self.assertGreater(DEFAULT.percentage(-1.0), 0)

    def test_the_display_keeps_a_decimal_at_the_top(self) -> None:
        self.assertNotEqual(DEFAULT.display(0.60), DEFAULT.display(0.75))

    def test_an_inverted_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SeparationCalibration(impostor_ceiling=0.6, genuine_floor=0.4)

    def test_describe_states_both_anchors(self) -> None:
        text = DEFAULT.describe()
        self.assertIn("0.200", text)
        self.assertIn("0.480", text)


class FitTests(unittest.TestCase):
    def test_the_anchors_come_from_the_measured_populations(self) -> None:
        genuine = [0.55, 0.60, 0.65, 0.70, 0.75]
        impostor = [0.00, 0.05, 0.10, 0.15, 0.20]

        calibration = fit(genuine, impostor)

        self.assertGreater(calibration.impostor_ceiling, 0.15)
        self.assertLess(calibration.impostor_ceiling, 0.25)
        self.assertGreater(calibration.genuine_floor, 0.50)
        self.assertLess(calibration.genuine_floor, 0.60)
        self.assertEqual(calibration.genuine_pairs, 5)
        self.assertEqual(calibration.impostor_pairs, 5)

    def test_overlapping_populations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit([0.4, 0.5], [0.45, 0.55])

    def test_an_empty_population_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit([], [0.1])
        with self.assertRaises(ValueError):
            fit([0.6], [])


class MeasurementTests(unittest.TestCase):
    def _gallery(self, count: int = 3) -> GalleryIndex:
        return GalleryIndex([identity(f"00{index}", seed=index) for index in range(1, count + 1)])

    def test_every_pair_is_scored(self) -> None:
        measurement = measure(self._gallery(3))

        assert measurement is not None
        # Three identities with three samples: 3 x 3 same-person and 3 x 9 different.
        self.assertEqual(len(measurement.genuine), 9)
        self.assertEqual(len(measurement.impostor), 27)

    def test_a_single_identity_cannot_be_measured(self) -> None:
        self.assertIsNone(measure(self._gallery(1)))

    def test_the_threshold_advice_names_the_rejected_pairs(self) -> None:
        measurement = SeparationMeasurement(
            genuine=(0.50, 0.60, 0.70), impostor=(0.05, 0.10)
        )

        advice = measurement.threshold_advice(0.65)

        self.assertIn("rejects 2", advice)
        self.assertIn("0.100", advice)

    def test_a_threshold_above_every_pair_is_reported_as_safe(self) -> None:
        measurement = SeparationMeasurement(genuine=(0.80, 0.90), impostor=(0.05,))

        self.assertIn("accepts every", measurement.threshold_advice(0.75))

    def test_describe_reports_both_populations(self) -> None:
        measurement = SeparationMeasurement(genuine=(0.5, 0.6), impostor=(0.1,))

        text = measurement.describe()

        self.assertIn("same person", text)
        self.assertIn("different people", text)


if __name__ == "__main__":
    unittest.main()
