from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from banhammer.messages import (
    DeduplicationError,
    DedupSettings,
    NearDuplicateFinder,
    NearDuplicates,
    shingles,
)

BASE = "the quick brown fox jumps over the lazy dog near the river bank"


def finder(
    jaccard: float = 0.8, shingle_size: int = 5, minhash_seed: int = 1
) -> NearDuplicateFinder:
    return NearDuplicateFinder(
        DedupSettings(
            shingle_size=shingle_size,
            num_perm=128,
            minhash_seed=minhash_seed,
            lsh_recall_weight=0.95,
            jaccard_threshold=jaccard,
            translation_threshold=0.85,
            candidate_floor=0.7,
            k=5,
            chunk_size=16,
            max_group_size_warning=50,
        )
    )


def jaccard(first: str, second: str, size: int = 5) -> float:
    a, b = shingles(first, size), shingles(second, size)
    return len(a & b) / len(a | b)


def test_shingles_are_character_ngrams_and_short_text_is_one_shingle() -> None:
    assert shingles("abcdefg", 5) == {"abcde", "bcdef", "cdefg"}
    assert shingles("abc", 5) == {"abc"}
    assert shingles("abcde", 5) == {"abcde"}
    assert shingles("aaaaaaa", 5) == {"aaaaa"}


def test_a_one_word_edit_of_a_long_message_is_a_near_duplicate() -> None:
    edited = BASE.replace("quick", "quack")
    assert jaccard(BASE, edited) >= 0.8
    assert finder().find(["a", "b"], [BASE, edited]).edges == [("a", "b")]


def test_unrelated_and_moderately_similar_messages_are_not_duplicates() -> None:
    moderate = "the quick brown fox jumps over a sleeping cat"
    assert jaccard(BASE, moderate) < 0.8
    other = "completely different words about something else entirely"
    assert finder().find(["a", "b", "c"], [BASE, moderate, other]).edges == []


def test_every_returned_edge_has_an_exact_jaccard_of_at_least_t1() -> None:
    variants = [BASE.replace("fox", word) for word in ("cat", "owl", "ape", "elk", "emu")]
    variants += [BASE[:n] for n in (30, 45, 55)]
    ids = [f"m{i}" for i in range(len(variants))]
    for threshold in (0.5, 0.8, 0.95):
        for first, second in finder(threshold).find(ids, variants).edges:
            assert jaccard(variants[ids.index(first)], variants[ids.index(second)]) >= threshold


def test_edges_are_unique_and_ordered_by_id() -> None:
    ids = ["c", "a", "b"]
    texts = [BASE, BASE.replace("dog", "dig"), BASE.replace("dog", "dug")]
    expected = sorted(
        (min(first, second), max(first, second))
        for (first, one), (second, two) in combinations(zip(ids, texts, strict=True), 2)
        if jaccard(one, two) >= 0.8
    )
    assert len(expected) == 3
    assert finder().find(ids, texts).edges == expected


def test_the_result_does_not_depend_on_the_input_order() -> None:
    texts = [BASE, BASE.replace("dog", "dig"), "unrelated chatter here and there", BASE + " ok"]
    ids = ["a", "b", "c", "d"]
    forward = finder().find(ids, texts).edges
    backward = finder().find(ids[::-1], texts[::-1]).edges
    assert forward == backward


def test_empty_and_single_inputs_give_no_edges() -> None:
    assert finder().find([], []).edges == []
    assert finder().find(["a"], [BASE]).edges == []


def test_mismatched_inputs_and_repeated_ids_are_rejected() -> None:
    with pytest.raises(DeduplicationError, match="in length"):
        finder().find(["a"], [BASE, BASE])
    with pytest.raises(DeduplicationError, match="unique"):
        finder().find(["a", "a"], [BASE, BASE + "x"])


@given(
    words=st.lists(
        st.sampled_from(["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]),
        min_size=6,
        max_size=14,
    ),
    swaps=st.integers(0, 3),
)
@settings(deadline=None, max_examples=40)
def test_no_false_positives_for_generated_variants(words: list[str], swaps: int) -> None:
    text = " ".join(words)
    variant = " ".join([*words[: len(words) - swaps], *["zzzzz"] * swaps])
    edges = finder().find(["a", "b"], [text, variant]).edges
    if edges:
        assert jaccard(text, variant) >= 0.8


def noisy_pairs(count: int, seed: int) -> list[tuple[str, str, float]]:
    """Pairs of random strings, the second a copy with a few characters replaced."""
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz "))
    pairs = []
    while len(pairs) < count:
        text = "".join(rng.choice(alphabet, size=int(rng.integers(60, 140))))
        edited = list(text)
        for position in rng.choice(len(text), size=int(rng.integers(1, 4)), replace=False):
            edited[position] = "#"
        variant = "".join(edited)
        similarity = jaccard(text, variant)
        if 0.8 <= similarity <= 0.9:
            pairs.append((text, variant, similarity))
    return pairs


def test_lsh_recall_at_true_jaccard_between_08_and_09_is_at_least_95_percent() -> None:
    pairs = noisy_pairs(300, seed=0)
    ids = [f"{side}{index:03d}" for index in range(len(pairs)) for side in "ab"]
    texts = [text for pair in pairs for text in pair[:2]]
    found = set(finder().find(ids, texts).edges)
    recall = sum((f"a{i:03d}", f"b{i:03d}") in found for i in range(len(pairs))) / len(pairs)
    assert recall >= 0.95


def test_the_candidates_count_what_lsh_proposed_and_the_edges_what_survived_the_check() -> None:
    near = BASE.replace("quick", "quack")
    moderate = BASE[:45]
    result = finder(0.9).find(["a", "b", "c"], [BASE, near, moderate])
    assert isinstance(result, NearDuplicates)
    assert result.candidates >= len(result.edges)
    assert {frozenset(edge) for edge in result.edges} <= {frozenset(("a", "b"))}


def test_the_minhash_seed_is_a_setting_and_does_not_change_which_pairs_are_verified() -> None:
    pairs = noisy_pairs(60, seed=1)
    ids = [f"{side}{index:03d}" for index in range(len(pairs)) for side in "ab"]
    texts = [text for pair in pairs for text in pair[:2]]
    first = finder(minhash_seed=1).find(ids, texts)
    again = finder(minhash_seed=1).find(ids, texts)
    other = finder(minhash_seed=2).find(ids, texts)
    assert first == again
    assert all(jaccard(texts[ids.index(a)], texts[ids.index(b)]) >= 0.8 for a, b in other.edges)
