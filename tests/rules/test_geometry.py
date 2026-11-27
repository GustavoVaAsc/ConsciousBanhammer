import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from banhammer.rules.geometry import cluster_diameter


def unit_rows(rows: list[list[float]]) -> NDArray[np.float32]:
    matrix = np.asarray(rows, dtype=np.float64)
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


nonzero_vector = st.lists(
    st.floats(min_value=-1, max_value=1, allow_nan=False), min_size=3, max_size=3
).filter(lambda v: sum(x * x for x in v) > 0.01)
matrices = st.lists(nonzero_vector, min_size=1, max_size=8).map(unit_rows)


def test_a_single_vector_has_diameter_zero() -> None:
    assert cluster_diameter(unit_rows([[1.0, 2.0, 3.0]])) == 0.0


def test_no_vectors_have_diameter_zero() -> None:
    assert cluster_diameter(np.empty((0, 3), dtype=np.float32)) == 0.0


def test_orthogonal_vectors_are_at_distance_one() -> None:
    assert cluster_diameter(unit_rows([[1, 0, 0], [0, 1, 0]])) == pytest.approx(1.0)


def test_opposite_vectors_are_at_the_maximum_distance_two() -> None:
    assert cluster_diameter(unit_rows([[1, 0, 0], [-1, 0, 0]])) == pytest.approx(2.0)


def test_the_diameter_is_the_largest_pair_not_the_neighbour_chain() -> None:
    chain = unit_rows([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
    assert cluster_diameter(chain) == pytest.approx(1.0)
    assert cluster_diameter(chain[:2]) == pytest.approx(1 - np.sqrt(0.5))


@given(matrices)
def test_it_equals_the_brute_force_maximum_over_pairs(vectors: NDArray[np.float32]) -> None:
    pairs = itertools.combinations(range(len(vectors)), 2)
    expected = max((1 - float(vectors[i] @ vectors[j]) for i, j in pairs), default=0.0)
    assert cluster_diameter(vectors) == pytest.approx(max(expected, 0.0), abs=1e-5)


@given(matrices, st.data())
def test_it_does_not_depend_on_the_order_of_the_rows(
    vectors: NDArray[np.float32], data: st.DataObject
) -> None:
    order = data.draw(st.permutations(range(len(vectors))))
    assert cluster_diameter(vectors[order]) == pytest.approx(cluster_diameter(vectors), abs=1e-6)


@given(matrices)
def test_it_lies_between_zero_and_two(vectors: NDArray[np.float32]) -> None:
    assert 0.0 <= cluster_diameter(vectors) <= 2.0


@given(matrices, matrices)
def test_adding_members_never_shrinks_the_diameter(
    vectors: NDArray[np.float32], more: NDArray[np.float32]
) -> None:
    assert cluster_diameter(np.vstack([vectors, more])) >= cluster_diameter(vectors) - 1e-6
