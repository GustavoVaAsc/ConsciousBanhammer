import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from banhammer.core.union_find import UnionFind
from banhammer.embeddings import ExactNeighbourSearch, NeighbourPairs, NeighbourSearchError

CPU = torch.device("cpu")


def normalized(rows: int, dimension: int, seed: int) -> NDArray[np.float32]:
    matrix = np.random.default_rng(seed).normal(size=(rows, dimension)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def search(k: int = 3, chunk_size: int = 4, min_score: float | None = None) -> ExactNeighbourSearch:
    return ExactNeighbourSearch(k=k, chunk_size=chunk_size, device=CPU, min_score=min_score)


def brute_force(
    queries: NDArray[np.float32],
    corpus: NDArray[np.float32],
    k: int,
    min_score: float | None = None,
    *,
    exclude_self: bool = False,
) -> NeighbourPairs:
    scores = queries @ corpus.T
    if exclude_self:
        np.fill_diagonal(scores, -np.inf)
    width = max(min(k, corpus.shape[0] - int(exclude_self)), 0)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :width]
    top = np.take_along_axis(scores, order, axis=1)
    rows = np.broadcast_to(np.arange(queries.shape[0])[:, None], (queries.shape[0], width))
    keep = top >= (-np.inf if min_score is None else min_score)
    return NeighbourPairs(rows[keep], order[keep], top[keep])


def assert_same_pairs(result: NeighbourPairs, expected: NeighbourPairs) -> None:
    np.testing.assert_array_equal(result.query_indices, expected.query_indices)
    np.testing.assert_array_equal(result.corpus_indices, expected.corpus_indices)
    np.testing.assert_allclose(result.scores, expected.scores, atol=1e-6)


def test_within_matches_a_brute_force_search() -> None:
    embeddings = normalized(40, 8, seed=0)
    result = search(k=5, chunk_size=7).within(embeddings)
    assert_same_pairs(result, brute_force(embeddings, embeddings, 5, exclude_self=True))
    assert len(result) == 200


def test_between_matches_a_brute_force_search_and_includes_equal_rows() -> None:
    queries = normalized(12, 8, seed=1)
    corpus = np.vstack([normalized(30, 8, seed=2), queries[:2]])
    result = search(k=4, chunk_size=5).between(queries, corpus)
    assert_same_pairs(result, brute_force(queries, corpus, 4))
    assert result.corpus_indices[result.query_indices == 0][0] == 30
    np.testing.assert_allclose(result.scores[result.query_indices == 0][0], 1.0, atol=1e-6)


engines = st.builds(
    search,
    k=st.integers(1, 20),
    chunk_size=st.integers(1, 20),
    min_score=st.one_of(st.none(), st.floats(-1.0, 1.0)),
)


@given(
    queries=st.integers(0, 15), corpus=st.integers(0, 15), engine=engines, seed=st.integers(0, 999)
)
@settings(deadline=None)
def test_between_equals_brute_force_for_any_shape_chunk_and_floor(
    queries: int, corpus: int, engine: ExactNeighbourSearch, seed: int
) -> None:
    query_matrix, corpus_matrix = normalized(queries, 6, seed), normalized(corpus, 6, seed + 1)
    expected = brute_force(query_matrix, corpus_matrix, engine.k, engine.min_score)
    assert_same_pairs(engine.between(query_matrix, corpus_matrix), expected)


@given(rows=st.integers(0, 15), engine=engines, seed=st.integers(0, 999))
@settings(deadline=None)
def test_within_equals_brute_force_and_never_pairs_a_row_with_itself(
    rows: int, engine: ExactNeighbourSearch, seed: int
) -> None:
    embeddings = normalized(rows, 6, seed)
    result = engine.within(embeddings)
    assert not (result.query_indices == result.corpus_indices).any()
    expected = brute_force(embeddings, embeddings, engine.k, engine.min_score, exclude_self=True)
    assert_same_pairs(result, expected)


def test_pairs_are_ordered_by_query_then_descending_score_and_typed() -> None:
    result = search(k=6, chunk_size=10).within(normalized(30, 5, seed=3))
    assert result.query_indices.dtype == np.int64
    assert result.corpus_indices.dtype == np.int64
    assert result.scores.dtype == np.float32
    same_query = np.diff(result.query_indices) == 0
    assert (np.diff(result.query_indices) >= 0).all()
    assert (np.diff(result.scores)[same_query] <= 0).all()


def test_edges_are_plain_int_tuples_in_pair_order() -> None:
    result = search(k=2, chunk_size=3).within(normalized(6, 4, seed=13))
    edges = result.edges()
    assert len(edges) == len(result) == 12
    assert all(type(first) is int and type(second) is int for first, second in edges)
    assert edges[0] == (int(result.query_indices[0]), int(result.corpus_indices[0]))
    assert len(search().within(normalized(1, 4, seed=14)).edges()) == 0


def test_within_edges_feed_union_all_to_group_near_duplicates() -> None:
    base = normalized(3, 8, seed=15)
    nudge = base[:2] + 1e-3
    nudge /= np.linalg.norm(nudge, axis=1, keepdims=True)
    embeddings = np.vstack([base, nudge]).astype(np.float32)
    result = search(k=1, chunk_size=2, min_score=0.99).within(embeddings)
    union_find = UnionFind(range(len(embeddings)))
    assert union_find.union_all(result.edges()) == 2
    assert union_find.groups() == {0: (0, 3), 1: (1, 4), 2: (2,)}


def test_min_score_keeps_only_pairs_at_or_above_the_floor() -> None:
    embeddings = normalized(30, 5, seed=4)
    everything = search(k=29, chunk_size=8).within(embeddings)
    floor = float(np.median(everything.scores))
    result = search(k=29, chunk_size=8, min_score=floor).within(embeddings)
    assert (result.scores >= floor).all()
    assert len(result) == int((everything.scores >= floor).sum())


def test_min_score_applies_within_the_top_k_so_rows_can_have_fewer_than_k_pairs() -> None:
    unit = np.eye(3, dtype=np.float32)
    result = search(k=2, chunk_size=2, min_score=0.5).within(unit)
    assert len(result) == 0
    result = search(k=2, chunk_size=2, min_score=0.0).within(unit)
    assert len(result) == 6


def test_ties_are_listed_by_ascending_index() -> None:
    unit = np.array([[1.0, 0.0]], dtype=np.float32)
    embeddings = np.vstack([unit, unit, unit, unit])
    result = search(k=3, chunk_size=4).within(embeddings)
    np.testing.assert_array_equal(result.corpus_indices[result.query_indices == 0], [1, 2, 3])
    np.testing.assert_array_equal(result.corpus_indices[result.query_indices == 3], [0, 1, 2])


def test_a_single_row_has_no_neighbours_within_but_has_some_between() -> None:
    row = normalized(1, 4, seed=5)
    assert len(search().within(row)) == 0
    assert len(search().between(row, normalized(3, 4, seed=6))) == 3
    assert len(search().between(row, normalized(0, 4, seed=6))) == 0


@pytest.mark.parametrize(("k", "chunk_size"), [(0, 4), (-1, 4), (3, 0), (3, -2)])
def test_non_positive_k_or_chunk_size_is_rejected(k: int, chunk_size: int) -> None:
    with pytest.raises(NeighbourSearchError, match="positive"):
        search(k, chunk_size)


@pytest.mark.parametrize("min_score", [float("nan"), float("inf")])
def test_a_non_finite_min_score_is_rejected(min_score: float) -> None:
    with pytest.raises(NeighbourSearchError, match="finite"):
        search(min_score=min_score)


def test_a_matrix_that_is_not_float32_two_dimensional_is_rejected() -> None:
    with pytest.raises(NeighbourSearchError, match="float32"):
        search().within(np.eye(3, dtype=np.float64))
    with pytest.raises(NeighbourSearchError, match="two-dimensional"):
        search().within(np.ones(3, dtype=np.float32))


def test_non_finite_and_non_normalized_rows_are_rejected() -> None:
    good = normalized(3, 4, seed=7)
    for name, bad_row in [
        ("NaN", [np.nan, 0, 0, 0]),
        ("infinity", [np.inf, 0, 0, 0]),
        ("L2-normalized", [0, 0, 0, 0]),
        ("L2-normalized", [2, 0, 0, 0]),
    ]:
        bad = np.vstack([good, np.array(bad_row, dtype=np.float32)])
        with pytest.raises(NeighbourSearchError, match=name):
            search().within(bad)
        with pytest.raises(NeighbourSearchError, match=name):
            search().between(good, bad)
        with pytest.raises(NeighbourSearchError, match=name):
            search().between(bad, good)


def test_queries_and_corpus_of_different_dimensions_are_rejected() -> None:
    with pytest.raises(NeighbourSearchError, match="columns"):
        search().between(normalized(3, 4, seed=8), normalized(3, 5, seed=9))


def test_non_contiguous_input_is_accepted_and_not_modified() -> None:
    embeddings = normalized(10, 8, seed=10)[:, ::2].copy(order="F")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    before = embeddings.copy()
    result = search(k=3, chunk_size=3).within(embeddings)
    assert_same_pairs(result, brute_force(embeddings, embeddings, 3, exclude_self=True))
    np.testing.assert_array_equal(embeddings, before)


def test_search_settings_are_frozen() -> None:
    with pytest.raises(AttributeError):
        search().k = 5  # type: ignore[misc]


@pytest.mark.gpu
def test_gpu_matches_cpu() -> None:
    queries, corpus = normalized(200, 16, seed=11), normalized(300, 16, seed=12)
    on_gpu = ExactNeighbourSearch(k=5, chunk_size=64, device=torch.device("cuda"), min_score=0.1)
    assert_same_pairs(on_gpu.between(queries, corpus), search(5, 64, 0.1).between(queries, corpus))
    assert_same_pairs(on_gpu.within(corpus), search(5, 64, 0.1).within(corpus))
