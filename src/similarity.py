"""Cosine similarity and the identity-matching decision.

Embeddings are always L2-normalised, so the dot product equals cosine similarity
and lives in [-1, 1]. Two rules keep false positives low:

* an identity is matched by its *best* sample, never by an average, so one bad
  angle cannot drag a correct identity below the threshold;
* a runner-up that is nearly as close as the winner rejects the match instead of
  guessing between two people.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

_CLIP_LIMIT = 1.0


class EmbeddingError(ValueError):
    """An embedding is empty, non-finite, or has an incompatible shape."""


def normalize_embedding(embedding: ArrayLike) -> np.ndarray:
    """Return the L2-normalised float32 vector, or raise ``EmbeddingError``."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise EmbeddingError("Embedding is empty.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise EmbeddingError("Embedding has zero or non-finite length.")
    return (vector / norm).astype(np.float32, copy=False)


def cosine_similarity(first: ArrayLike, second: ArrayLike) -> float:
    """Cosine similarity of two vectors, independent of their magnitudes."""
    left = normalize_embedding(first)
    right = normalize_embedding(second)
    if left.size != right.size:
        raise EmbeddingError(
            f"Embedding sizes differ ({left.size} vs {right.size}); "
            "both must come from the same model."
        )
    return float(np.clip(np.dot(left, right), -_CLIP_LIMIT, _CLIP_LIMIT))


def cosine_similarity_matrix(query: ArrayLike, gallery: ArrayLike) -> np.ndarray:
    """Cosine similarity of one vector against every row of a matrix."""
    vector = normalize_embedding(query)
    matrix = np.asarray(gallery, dtype=np.float32)
    if matrix.ndim == 1 and matrix.size:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise EmbeddingError("Gallery is empty; nothing to compare against.")
    if matrix.shape[1] != vector.size:
        raise EmbeddingError(
            f"Gallery rows have {matrix.shape[1]} values but the query has {vector.size}."
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise EmbeddingError("Gallery contains a zero or non-finite embedding.")
    scores = (matrix / norms) @ vector
    return np.clip(scores, -_CLIP_LIMIT, _CLIP_LIMIT).astype(np.float32, copy=False)


def to_percentage(similarity: float) -> int:
    """Map a similarity in [-1, 1] to a 0-100 integer for display.

    Similarity is not a probability. Showing it as a percentage is a readability
    convention; the README states this explicitly so the number is not misread.
    """
    return round(float(np.clip(similarity, 0.0, 1.0)) * 100.0)


@dataclass(frozen=True)
class IdentityRecord:
    """One registered identity with all of its sample embeddings."""

    employee_id: str
    full_name: str
    embeddings: np.ndarray  # (samples, dimension), float32


class GalleryIndex:
    """Flattened view of every sample, prepared for fast comparison."""

    def __init__(self, identities: Iterable[IdentityRecord]) -> None:
        records = list(identities)
        self._names: dict[str, str] = {}
        self._index_by_id: dict[str, int] = {}
        if not records:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._owner_indices = np.zeros((0,), dtype=np.int32)
            self._identity_ids: tuple[str, ...] = ()
            return

        rows: list[np.ndarray] = []
        owners: list[int] = []
        for owner_index, record in enumerate(records):
            self._names[record.employee_id] = record.full_name
            self._index_by_id[record.employee_id] = owner_index
            for row in record.embeddings:
                rows.append(normalize_embedding(row))
                owners.append(owner_index)

        dimensions = {row.size for row in rows}
        if len(dimensions) != 1:
            raise EmbeddingError(f"Gallery mixes embedding sizes: {sorted(dimensions)}.")
        self._matrix = np.stack(rows).astype(np.float32)
        self._owner_indices = np.asarray(owners, dtype=np.int32)
        self._identity_ids = tuple(record.employee_id for record in records)

    @property
    def is_empty(self) -> bool:
        return self._matrix.shape[0] == 0

    @property
    def identity_ids(self) -> tuple[str, ...]:
        return self._identity_ids

    @property
    def identity_count(self) -> int:
        return len(self._identity_ids)

    @property
    def sample_count(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1]) if self._matrix.size else 0

    def name_for(self, employee_id: str) -> str | None:
        return self._names.get(employee_id)

    def matches_for(self, employee_id: str) -> np.ndarray | None:
        """All normalised embeddings of one identity, or None if unknown."""
        owner_index = self._index_by_id.get(employee_id)
        if owner_index is None:
            return None
        mask = self._owner_indices == owner_index
        return self._matrix[mask] if np.any(mask) else None

    def best_similarity_per_identity(self, embedding: ArrayLike) -> dict[str, float]:
        """Highest similarity for each identity in the gallery."""
        if self.is_empty:
            return {}
        scores = cosine_similarity_matrix(embedding, self._matrix)
        best: dict[str, float] = {}
        for owner_index, employee_id in enumerate(self._identity_ids):
            mask = self._owner_indices == owner_index
            if np.any(mask):
                best[employee_id] = float(np.max(scores[mask]))
        return best


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching one embedding against the whole gallery."""

    employee_id: str | None
    full_name: str | None
    similarity: float
    accepted: bool
    reason: str

    @property
    def display_name(self) -> str:
        """Name safe to show; "Unknown" unless the match was accepted."""
        return self.full_name if self.accepted and self.full_name else "Unknown"

    @property
    def display_id(self) -> str:
        return self.employee_id if self.accepted and self.employee_id else "UNKNOWN"

    @property
    def percentage(self) -> int:
        return to_percentage(self.similarity)


class FaceMatcher:
    """Threshold plus ambiguity margin, applied to a gallery index."""

    def __init__(self, threshold: float, ambiguity_margin: float = 0.0) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [-1, 1], got {threshold}.")
        if not 0.0 <= ambiguity_margin <= 1.0:
            raise ValueError(f"ambiguity_margin must be within [0, 1], got {ambiguity_margin}.")
        self._threshold = float(threshold)
        self._margin = float(ambiguity_margin)

    @property
    def threshold(self) -> float:
        return self._threshold

    def match(self, embedding: ArrayLike, gallery: GalleryIndex) -> MatchResult:
        if gallery.is_empty:
            return MatchResult(None, None, 0.0, False, "No identities are registered yet.")

        scores = gallery.best_similarity_per_identity(embedding)
        if not scores:
            return MatchResult(None, None, 0.0, False, "Gallery has no usable samples.")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else float("-inf")

        if best_score < self._threshold:
            return MatchResult(
                None,
                None,
                best_score,
                False,
                f"Best similarity {best_score:.3f} is below the threshold {self._threshold:.3f}.",
            )

        if (
            self._margin > 0.0
            and runner_up > float("-inf")
            and best_score - runner_up < self._margin
        ):
            return MatchResult(
                None,
                None,
                best_score,
                False,
                f"Two identities are too close ({best_score:.3f} vs {runner_up:.3f}, "
                f"gap below margin {self._margin:.3f}).",
            )

        return MatchResult(
            best_id,
            gallery.name_for(best_id),
            best_score,
            True,
            f"Matched at similarity {best_score:.3f} (threshold {self._threshold:.3f}).",
        )


def identity_records_from_mapping(
    mapping: Mapping[str, list[np.ndarray]],
) -> list[IdentityRecord]:
    """Build records from ``{employee_id: [embedding, ...]}`` for tests and imports."""
    return [
        IdentityRecord(employee_id=str(key), full_name=str(key), embeddings=np.stack(rows))
        for key, rows in mapping.items()
        if rows
    ]
