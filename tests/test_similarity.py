"""Similarity maths, the gallery index, and the matching decision."""

from __future__ import annotations

import unittest

import numpy as np

from src.similarity import (
    EmbeddingError,
    FaceMatcher,
    GalleryIndex,
    IdentityRecord,
    cosine_similarity,
    cosine_similarity_matrix,
    identity_records_from_mapping,
    normalize_embedding,
    to_percentage,
)
from tests.support import unit_vector


def gallery(*identities: tuple[str, str, int]) -> GalleryIndex:
    records = [
        IdentityRecord(employee_id, name, np.stack([unit_vector(seed + index) for index in range(3)]))
        for seed, (employee_id, name, _unused) in enumerate(identities)
    ]
    return GalleryIndex(records)


class NormaliseTests(unittest.TestCase):
    def test_length_is_one(self) -> None:
        vector = normalize_embedding([3.0, 4.0])
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_zero_vector_is_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            normalize_embedding([0.0, 0.0, 0.0])

    def test_empty_vector_is_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            normalize_embedding([])

    def test_non_finite_vector_is_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            normalize_embedding([1.0, float("nan")])


class CosineTests(unittest.TestCase):
    def test_identical_vectors_score_one(self) -> None:
        vector = unit_vector(3)
        self.assertAlmostEqual(cosine_similarity(vector, vector), 1.0, places=6)

    def test_opposite_vectors_score_minus_one(self) -> None:
        vector = unit_vector(3)
        self.assertAlmostEqual(cosine_similarity(vector, -vector), -1.0, places=6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_mismatched_sizes_are_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_matrix_matches_the_pairwise_result(self) -> None:
        query = unit_vector(1)
        matrix = np.stack([unit_vector(2), unit_vector(3)])
        scores = cosine_similarity_matrix(query, matrix)
        self.assertEqual(scores.shape, (2,))
        self.assertAlmostEqual(float(scores[0]), cosine_similarity(query, matrix[0]), places=5)

    def test_empty_matrix_is_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            cosine_similarity_matrix(unit_vector(1), np.zeros((0, 8), dtype=np.float32))


class PercentageTests(unittest.TestCase):
    def test_scale_is_zero_to_one_hundred(self) -> None:
        self.assertEqual(to_percentage(0.0), 0)
        self.assertEqual(to_percentage(0.455), 46)
        self.assertEqual(to_percentage(1.0), 100)

    def test_negative_similarity_clamps_to_zero(self) -> None:
        self.assertEqual(to_percentage(-0.4), 0)

    def test_above_one_clamps_to_one_hundred(self) -> None:
        self.assertEqual(to_percentage(1.4), 100)


class GalleryTests(unittest.TestCase):
    def test_counts(self) -> None:
        index = gallery(("001", "Alice", 1), ("002", "Bob", 2))
        self.assertEqual(index.identity_count, 2)
        self.assertEqual(index.sample_count, 6)
        self.assertFalse(index.is_empty)

    def test_best_similarity_per_identity(self) -> None:
        index = gallery(("001", "Alice", 1))
        scores = index.best_similarity_per_identity(unit_vector(0))
        self.assertEqual(set(scores), {"001"})
        self.assertGreater(scores["001"], 0.9)

    def test_name_lookup(self) -> None:
        index = gallery(("007", "Bond", 1))
        self.assertEqual(index.name_for("007"), "Bond")
        self.assertIsNone(index.name_for("404"))

    def test_matches_for_returns_all_rows(self) -> None:
        index = gallery(("001", "Alice", 1))
        rows = index.matches_for("001")
        assert rows is not None
        self.assertEqual(rows.shape[0], 3)

    def test_empty_gallery(self) -> None:
        index = GalleryIndex([])
        self.assertTrue(index.is_empty)
        self.assertEqual(index.best_similarity_per_identity(unit_vector(1)), {})

    def test_mixed_dimensions_are_rejected(self) -> None:
        records = [
            IdentityRecord("001", "A", np.ones((2, 4), dtype=np.float32)),
            IdentityRecord("002", "B", np.ones((2, 5), dtype=np.float32)),
        ]
        with self.assertRaises(EmbeddingError):
            GalleryIndex(records)

    def test_records_from_mapping(self) -> None:
        records = identity_records_from_mapping({"001": [unit_vector(1), unit_vector(2)]})
        self.assertEqual(records[0].employee_id, "001")
        self.assertEqual(records[0].embeddings.shape[0], 2)


class MatcherTests(unittest.TestCase):
    def test_clear_match_is_accepted(self) -> None:
        index = gallery(("001", "Alice", 1))
        result = FaceMatcher(0.3).match(unit_vector(0), index)
        self.assertTrue(result.accepted)
        self.assertEqual(result.employee_id, "001")
        self.assertEqual(result.display_name, "Alice")

    def test_poor_match_is_rejected(self) -> None:
        index = gallery(("001", "Alice", 1))
        result = FaceMatcher(0.999).match(unit_vector(40), index)
        self.assertFalse(result.accepted)
        self.assertEqual(result.display_name, "Unknown")
        self.assertEqual(result.display_id, "UNKNOWN")

    def test_empty_gallery_never_matches(self) -> None:
        result = FaceMatcher(0.1).match(unit_vector(1), GalleryIndex([]))
        self.assertFalse(result.accepted)
        self.assertIn("no identities", result.reason.lower())

    def test_ambiguity_margin_rejects_close_calls(self) -> None:
        vector = unit_vector(5)
        records = [
            IdentityRecord("001", "Alice", vector.reshape(1, -1)),
            IdentityRecord("002", "Bob", (vector * 0.999).reshape(1, -1)),
        ]
        result = FaceMatcher(0.3, ambiguity_margin=0.5).match(vector, GalleryIndex(records))
        self.assertFalse(result.accepted)
        self.assertIn("too close", result.reason.lower())

    def test_percentage_is_exposed(self) -> None:
        index = gallery(("001", "Alice", 1))
        result = FaceMatcher(0.3).match(unit_vector(0), index)
        self.assertEqual(result.percentage, to_percentage(result.similarity))

    def test_threshold_bounds_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            FaceMatcher(1.5)
        with self.assertRaises(ValueError):
            FaceMatcher(0.5, ambiguity_margin=-1.0)


if __name__ == "__main__":
    unittest.main()
