import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from banhammer.core.errors import BanhammerError, InvalidKeyError, UnknownKeyError
from banhammer.core.union_find import UnionFind

edges_strategy = st.lists(
    st.tuples(st.text("abcdef", min_size=1, max_size=2), st.text("abcdef", min_size=1, max_size=2))
)


def test_new_keys_are_singleton_groups() -> None:
    union_find = UnionFind(["b", "a"])
    assert len(union_find) == 2
    assert union_find.group_count == 2
    assert "a" in union_find
    assert "z" not in union_find
    assert not union_find.connected("a", "b")


def test_union_merges_and_reports_whether_it_did() -> None:
    union_find = UnionFind(["a", "b", "c"])
    assert union_find.union("a", "b") is True
    assert union_find.union("b", "a") is False
    assert union_find.connected("a", "b")
    assert not union_find.connected("a", "c")
    assert union_find.group_count == 2


def test_union_is_transitive() -> None:
    union_find = UnionFind[str]()
    union_find.union("a", "b")
    union_find.union("c", "d")
    union_find.union("b", "c")
    assert union_find.connected("a", "d")
    assert union_find.group_count == 1


def test_union_adds_keys_it_has_not_seen() -> None:
    union_find = UnionFind[str]()
    union_find.union("x", "y")
    assert len(union_find) == 2
    assert union_find.connected("x", "y")


def test_add_is_idempotent_and_keeps_existing_groups() -> None:
    union_find = UnionFind(["a", "b"])
    union_find.union("a", "b")
    union_find.add("a")
    union_find.add("c")
    assert union_find.connected("a", "b")
    assert union_find.group_count == 2
    assert len(union_find) == 3


def test_self_union_changes_nothing() -> None:
    union_find = UnionFind(["a"])
    assert union_find.union("a", "a") is False
    assert union_find.group_count == 1


def test_group_id_is_the_smallest_member_whatever_the_merge_order() -> None:
    union_find = UnionFind(["m", "z", "a", "q"])
    union_find.union("z", "q")
    union_find.union("m", "z")
    assert union_find.group_id("q") == "m"
    union_find.union("a", "q")
    assert {union_find.group_id(key) for key in "mzaq"} == {"a"}


def test_group_ids_and_groups_cover_every_key() -> None:
    union_find = UnionFind(["b", "a", "d", "c", "e"])
    union_find.union("b", "a")
    union_find.union("d", "e")
    assert union_find.group_ids() == {"a": "a", "b": "a", "c": "c", "d": "d", "e": "d"}
    assert union_find.groups() == {"a": ("a", "b"), "c": ("c",), "d": ("d", "e")}


def test_integer_keys_order_numerically() -> None:
    union_find = UnionFind[int]()
    union_find.union(10, 9)
    assert union_find.group_id(10) == 9


def test_group_ids_do_not_change_when_a_group_grows_incrementally() -> None:
    union_find = UnionFind(["a", "b", "c"])
    union_find.union("b", "c")
    assert union_find.group_id("c") == "b"
    union_find.union("c", "a")
    assert union_find.group_id("c") == "a"


def test_unknown_key_raises_a_banhammer_error() -> None:
    union_find = UnionFind(["a"])
    for call in (
        lambda: union_find.find("b"),
        lambda: union_find.group_id("b"),
        lambda: union_find.connected("a", "b"),
    ):
        with pytest.raises(UnknownKeyError, match="'b'"):
            call()
    assert issubclass(UnknownKeyError, BanhammerError)


def test_find_survives_a_long_chain_without_recursion() -> None:
    size = 20_000
    union_find = UnionFind[int]()
    for key in range(size - 1):
        union_find.union(key, key + 1)
    assert union_find.group_count == 1
    assert union_find.group_id(size - 1) == 0


def test_group_size_counts_the_members_of_the_group() -> None:
    union_find = UnionFind(["a", "b", "c", "d"])
    union_find.union("a", "b")
    union_find.union("c", "b")
    assert union_find.group_size("a") == 3
    assert union_find.group_size("d") == 1
    with pytest.raises(UnknownKeyError):
        union_find.group_size("z")


def test_union_all_returns_the_number_of_merges() -> None:
    union_find = UnionFind[str]()
    merges = union_find.union_all([("a", "b"), ("b", "c"), ("c", "a"), ("d", "d"), ("x", "y")])
    assert merges == 3
    assert union_find.group_count == 3
    assert union_find.union_all([]) == 0


def test_union_all_accepts_a_generator() -> None:
    union_find = UnionFind[int]()
    assert union_find.union_all((n, n + 1) for n in range(4)) == 4
    assert union_find.group_size(0) == 5


@pytest.mark.parametrize("key", [np.int64(3), np.str_("a"), True, 1.5, None, ("a",), b"a"])
def test_a_key_that_is_not_a_plain_str_or_int_is_rejected(key: object) -> None:
    union_find = UnionFind[str]()
    with pytest.raises(InvalidKeyError):
        union_find.add(key)  # type: ignore[arg-type]
    assert len(union_find) == 0


def test_a_key_of_another_kind_than_the_first_is_rejected_without_changing_anything() -> None:
    union_find = UnionFind(["a"])
    with pytest.raises(InvalidKeyError, match="1"):
        union_find.add(1)  # type: ignore[arg-type]
    with pytest.raises(InvalidKeyError):
        union_find.union("a", 1)  # type: ignore[arg-type]
    assert len(union_find) == 1
    assert union_find.group_count == 1


def test_a_union_of_two_kinds_on_an_empty_instance_adds_nothing() -> None:
    union_find = UnionFind[str]()
    with pytest.raises(InvalidKeyError):
        union_find.union("a", 1)  # type: ignore[arg-type]
    assert len(union_find) == 0
    with pytest.raises(InvalidKeyError):
        union_find.union_all([("a", "b"), ("c", 2)])  # type: ignore[list-item]
    assert len(union_find) == 0


def test_invalid_key_error_is_a_banhammer_error() -> None:
    assert issubclass(InvalidKeyError, BanhammerError)


@given(edges=edges_strategy, seed=st.randoms(use_true_random=False))
def test_group_ids_are_independent_of_edge_order_and_direction(
    edges: list[tuple[str, str]], seed: random.Random
) -> None:
    shuffled = [(b, a) if seed.random() < 0.5 else (a, b) for a, b in edges]
    seed.shuffle(shuffled)
    first, second = UnionFind[str](), UnionFind[str]()
    for a, b in edges:
        first.union(a, b)
    for a, b in shuffled:
        second.union(a, b)
    assert first.group_ids() == second.group_ids()


@given(edges=edges_strategy)
def test_group_id_is_the_minimum_of_its_group_and_groups_partition_the_keys(
    edges: list[tuple[str, str]],
) -> None:
    union_find = UnionFind[str]()
    for a, b in edges:
        union_find.union(a, b)
    groups = union_find.groups()
    assert union_find.group_count == len(groups)
    assert sum(len(members) for members in groups.values()) == len(union_find)
    for group_id, members in groups.items():
        assert group_id == min(members) == members[0]
        assert all(union_find.group_id(member) == group_id for member in members)
