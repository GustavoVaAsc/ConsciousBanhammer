"""Exact top-k neighbours by dot product over L2-normalized embeddings, in row chunks.

For L2-normalized rows the dot product is the cosine similarity. Only one chunk of the
`queries x corpus` score matrix exists at a time, so memory is `chunk_size x corpus rows`
floats on the device plus the two embedding matrices.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from banhammer.embeddings.constants import EMBEDDING_MATRIX_NDIM, UNIT_NORM_TOLERANCE
from banhammer.embeddings.errors import NeighbourSearchError


@dataclass(frozen=True)
class NeighbourPairs:
    """Neighbour pairs as three parallel arrays, one entry per pair.

    Pairs are ordered by query row, then by descending score, then by ascending corpus row.
    `query_indices` and `corpus_indices` index the rows of the matrices that were searched.
    """

    query_indices: NDArray[np.int64]
    corpus_indices: NDArray[np.int64]
    scores: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.scores)

    def edges(self) -> list[tuple[int, int]]:
        """The pairs as `(query row, corpus row)` tuples of plain `int`, in the same order.

        Plain ints because `UnionFind` rejects NumPy integer keys; for `within` results the
        tuples feed `UnionFind.union_all` directly.
        """
        return list(zip(self.query_indices.tolist(), self.corpus_indices.tolist(), strict=True))


@dataclass(frozen=True)
class ExactNeighbourSearch:
    """Exact top-k search with a score floor, run in chunks of query rows on one device.

    Which of several corpus rows with exactly equal scores fills the last of the `k` places is
    up to the backend (`torch.topk`) and can differ between CPU and GPU; the order of the
    kept pairs is deterministic. A caller that needs exact-duplicate grouping must hash-dedup
    exact copies before searching instead of relying on `k` to cover them.

    Args:
        k: Neighbours kept per query row.
        chunk_size: Query rows scored per matrix product, which bounds the memory used.
        device: Where the products run.
        min_score: When given, pairs scoring below it are dropped, so a row can have fewer than
            `k` pairs. Among the `k` best, those at or above the floor are kept.
    """

    k: int
    chunk_size: int
    device: torch.device
    min_score: float | None = None

    def __post_init__(self) -> None:
        """Reject a non-positive `k` or `chunk_size` and a non-finite `min_score`."""
        if self.k < 1 or self.chunk_size < 1:
            message = (
                f"k and chunk_size must be positive, got k={self.k}, chunk_size={self.chunk_size}"
            )
            raise NeighbourSearchError(message)
        if self.min_score is not None and not math.isfinite(self.min_score):
            message = f"min_score must be finite, got {self.min_score}"
            raise NeighbourSearchError(message)

    def between(self, queries: NDArray[np.float32], corpus: NDArray[np.float32]) -> NeighbourPairs:
        """Top-k corpus rows for every query row, e.g. English queries against a Spanish corpus.

        Raises:
            NeighbourSearchError: A matrix is not two-dimensional, float32, finite and
                L2-normalized, or the two differ in dimension.
        """
        _validate_matrix("queries", queries)
        _validate_matrix("corpus", corpus)
        if queries.shape[1] != corpus.shape[1]:
            message = f"queries have {queries.shape[1]} columns, corpus has {corpus.shape[1]}"
            raise NeighbourSearchError(message)
        return self._search(queries, corpus, exclude_self=False)

    def within(self, embeddings: NDArray[np.float32]) -> NeighbourPairs:
        """Top-k neighbours of every row among the other rows, never the row itself.

        A row is excluded by position, so an identical row elsewhere is a neighbour with
        score 1.

        Raises:
            NeighbourSearchError: `embeddings` is not two-dimensional, float32, finite and
                L2-normalized.
        """
        _validate_matrix("embeddings", embeddings)
        return self._search(embeddings, embeddings, exclude_self=True)

    def _search(
        self, queries: NDArray[np.float32], corpus: NDArray[np.float32], *, exclude_self: bool
    ) -> NeighbourPairs:
        width = min(self.k, corpus.shape[0] - int(exclude_self))
        if width < 1 or queries.shape[0] == 0:
            return _no_pairs()
        with torch.inference_mode():
            corpus_tensor = torch.from_numpy(np.ascontiguousarray(corpus)).to(self.device)
            query_tensor = torch.from_numpy(np.ascontiguousarray(queries)).to(self.device)
            chunks = [
                self._chunk_pairs(
                    self._chunk_scores(
                        query_tensor, corpus_tensor, start, exclude_self=exclude_self
                    ),
                    start,
                    width,
                )
                for start in range(0, queries.shape[0], self.chunk_size)
            ]
        return _ordered(
            np.concatenate([chunk.query_indices for chunk in chunks]),
            np.concatenate([chunk.corpus_indices for chunk in chunks]),
            np.concatenate([chunk.scores for chunk in chunks]),
        )

    def _chunk_pairs(self, scores: torch.Tensor, start: int, width: int) -> NeighbourPairs:
        top = torch.topk(scores, width, dim=1)
        floor = -math.inf if self.min_score is None else self.min_score
        rows, slots = torch.nonzero(top.values >= floor, as_tuple=True)
        return NeighbourPairs(
            (rows + start).cpu().numpy(),
            top.indices[rows, slots].cpu().numpy(),
            top.values[rows, slots].cpu().numpy(),
        )

    def _chunk_scores(
        self, queries: torch.Tensor, corpus: torch.Tensor, start: int, *, exclude_self: bool
    ) -> torch.Tensor:
        chunk = queries[start : start + self.chunk_size]
        scores = chunk @ corpus.T
        if exclude_self:
            own_rows = torch.arange(chunk.shape[0], device=scores.device)
            scores[own_rows, own_rows + start] = -torch.inf
        return scores


def _no_pairs() -> NeighbourPairs:
    return NeighbourPairs(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))


def _ordered(
    query_indices: NDArray[np.int64], corpus_indices: NDArray[np.int64], scores: NDArray[np.float32]
) -> NeighbourPairs:
    order = np.lexsort((corpus_indices, -scores, query_indices))
    return NeighbourPairs(query_indices[order], corpus_indices[order], scores[order])


def _validate_matrix(name: str, matrix: NDArray[np.float32]) -> None:
    if matrix.ndim != EMBEDDING_MATRIX_NDIM:
        message = f"{name} must be two-dimensional, got {matrix.ndim} dimensions"
        raise NeighbourSearchError(message)
    if matrix.dtype != np.float32:
        message = f"{name} must be float32, got {matrix.dtype}"
        raise NeighbourSearchError(message)
    if not np.isfinite(matrix).all():
        message = f"{name} must not contain NaN or infinity"
        raise NeighbourSearchError(message)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=UNIT_NORM_TOLERANCE, rtol=0):
        message = f"{name} rows must be L2-normalized to within {UNIT_NORM_TOLERANCE}"
        raise NeighbourSearchError(message)
