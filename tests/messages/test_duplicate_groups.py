import logging

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from banhammer.messages import (
    DeduplicationError,
    DuplicateGrouper,
    NearDuplicateFinder,
    TranslationCandidateFinder,
)
from dedup_corpus import (
    E1,
    E1_EDITED,
    E1_PARAPHRASE,
    E1_SHOUTED,
    E2,
    E3,
    S1,
    S1_EDITED,
    S2,
    S3,
    VECTORS,
    dedup_settings,
    embedder,
    search,
)
from embedder_double import FakeEmbedder


def grouper(model: FakeEmbedder | None = None, **overrides: object) -> DuplicateGrouper:
    config = dedup_settings(**overrides)
    near = NearDuplicateFinder(config)
    translations = TranslationCandidateFinder(
        model or embedder(), search(config.k, config.candidate_floor)
    )
    return DuplicateGrouper(near, translations, config)


def messages(*rows: tuple[str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"message_id": pl.String, "text": pl.String, "language": pl.String},
        orient="row",
    )


def group_ids(frame: pl.DataFrame, **overrides: object) -> dict[str, str]:
    groups = grouper(**overrides).group(frame).groups
    return dict(zip(groups["message_id"], groups["dup_group_id"], strict=True))


def test_exact_copies_up_to_case_and_noise_share_a_group_named_after_the_smallest_id() -> None:
    frame = messages(
        ("m3", E1, "en"),
        ("m1", E1_SHOUTED, "en"),
        ("m2", f"{E1} https://example.com 😂", "en"),
        ("m4", E3, "en"),
    )
    assert group_ids(frame) == {"m3": "m1", "m1": "m1", "m2": "m1", "m4": "m4"}


def test_near_duplicates_of_the_same_language_share_a_group() -> None:
    frame = messages(
        ("a", E1, "en"),
        ("b", E1_EDITED, "en"),
        ("c", E3, "en"),
        ("d", S1, "es"),
        ("e", S1_EDITED, "es"),
    )
    ids = group_ids(frame, translation_threshold=1.0, candidate_floor=1.0)
    assert ids["a"] == ids["b"] == "a"
    assert ids["d"] == ids["e"] == "d"
    assert ids["c"] == "c"


def test_translations_above_t2_share_a_group_and_those_below_do_not() -> None:
    frame = messages(
        ("e1", E1, "en"), ("s1", S1, "es"), ("e2", E2, "en"), ("s2", S2, "es"), ("s3", S3, "es")
    )
    assert group_ids(frame) == {"e1": "e1", "s1": "e1", "e2": "e2", "s2": "e2", "s3": "s3"}
    stricter = group_ids(frame, translation_threshold=0.95)
    assert stricter["e1"] == "e1" and stricter["s1"] == "e1"
    assert stricter["e2"] == "e2" and stricter["s2"] == "s2"


def test_same_language_paraphrases_stay_apart_however_close_their_embeddings() -> None:
    frame = messages(("a", E1, "en"), ("b", E1_PARAPHRASE, "en"))
    assert group_ids(frame, translation_threshold=0.5, candidate_floor=0.5) == {"a": "a", "b": "b"}


def test_a_mixed_chain_of_edges_forms_one_group() -> None:
    frame = messages(
        ("e1", E1, "en"), ("e1b", E1_SHOUTED, "en"), ("s1", S1, "es"), ("s1b", S1_EDITED, "es")
    )
    assert set(group_ids(frame).values()) == {"e1"}


def test_exact_copies_are_embedded_once_so_they_cannot_crowd_out_the_top_k() -> None:
    model = embedder()
    frame = messages(
        ("e1", E1, "en"),
        ("e2", E1_SHOUTED, "en"),
        ("s1", S1, "es"),
        ("s2", S1, "es"),
        ("s3", S1, "es"),
        ("s4", S3, "es"),
    )
    ids = group_ids_with(model, frame, k=1)
    assert len({ids[m] for m in ("e1", "e2", "s1", "s2", "s3")}) == 1
    assert sorted(len(call) for call in model.calls) == [1, 2]


def group_ids_with(model: FakeEmbedder, frame: pl.DataFrame, **overrides: object) -> dict[str, str]:
    groups = grouper(model, **overrides).group(frame).groups
    return dict(zip(groups["message_id"], groups["dup_group_id"], strict=True))


def test_the_candidates_include_pairs_between_the_floor_and_t2_for_calibration() -> None:
    frame = messages(("e1", E1, "en"), ("s1", S1, "es"), ("e2", E2, "en"), ("s2", S2, "es"))
    result = grouper(translation_threshold=0.95, candidate_floor=0.8).group(frame)
    assert sorted(
        zip(result.candidates["en_message_id"], result.candidates["es_message_id"], strict=True)
    ) == [("e1", "s1"), ("e2", "s2")]
    assert result.edges.translation == 1


def test_edge_counts_describe_each_kind_of_edge() -> None:
    frame = messages(
        ("a", E1, "en"),
        ("b", E1_SHOUTED, "en"),
        ("c", E1_EDITED, "en"),
        ("d", S1, "es"),
        ("e", E3, "en"),
    )
    result = grouper().group(frame)
    assert (result.edges.exact, result.edges.near, result.edges.translation) == (1, 1, 2)


def test_a_threshold_met_exactly_counts_as_a_duplicate() -> None:
    echo = "saludos cordiales desde la oficina central"
    model = FakeEmbedder(VECTORS | {echo: [1, 0, 0, 0]})
    frame = messages(("e", E1, "en"), ("s", echo, "es"))
    ids = group_ids_with(model, frame, translation_threshold=1.0, candidate_floor=1.0)
    assert ids["e"] == ids["s"]


def test_output_follows_the_input_order_and_covers_every_message() -> None:
    frame = messages(("z", E3, "en"), ("a", E2, "en"), ("m", S3, "es"))
    result = grouper().group(frame)
    assert result.groups["message_id"].to_list() == ["z", "a", "m"]
    assert result.groups.columns == ["message_id", "dup_group_id"]


def test_one_language_only_and_empty_input_work() -> None:
    assert group_ids(messages(("a", E1, "en"), ("b", E1_SHOUTED, "en"))) == {"a": "a", "b": "a"}
    assert grouper().group(messages()).groups.height == 0


def test_repeated_message_ids_are_rejected() -> None:
    with pytest.raises(DeduplicationError, match="unique"):
        grouper().group(messages(("a", E1, "en"), ("a", E3, "en")))


POOL = [
    (E1, "en"),
    (E1_SHOUTED, "en"),
    (E1_EDITED, "en"),
    (E1_PARAPHRASE, "en"),
    (E2, "en"),
    (E3, "en"),
    (S1, "es"),
    (S1_EDITED, "es"),
    (S2, "es"),
    (S3, "es"),
]


@given(
    picks=st.lists(st.integers(0, len(POOL) - 1), min_size=0, max_size=14),
    seed=st.integers(0, 100),
)
@settings(deadline=None, max_examples=60)
def test_group_ids_do_not_depend_on_the_order_of_the_messages(picks: list[int], seed: int) -> None:
    rows = [(f"m{i:02d}", *POOL[pick]) for i, pick in enumerate(picks)]
    shuffled = [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))]
    assert group_ids(messages(*rows)) == group_ids(messages(*shuffled))


@given(picks=st.lists(st.integers(0, len(POOL) - 1), min_size=1, max_size=14))
@settings(deadline=None, max_examples=60)
def test_a_group_id_is_its_smallest_member_and_equal_texts_never_split(picks: list[int]) -> None:
    rows = [(f"m{i:02d}", *POOL[pick]) for i, pick in enumerate(picks)]
    ids = group_ids(messages(*rows))
    for message_id, group in ids.items():
        assert group <= message_id
        assert ids[group] == group
    by_text: dict[str, set[str]] = {}
    for message_id, text, _ in rows:
        by_text.setdefault(text.casefold(), set()).add(ids[message_id])
    assert all(len(groups) == 1 for groups in by_text.values())


def test_group_sizes_and_the_near_candidate_count_are_reported() -> None:
    frame = messages(
        ("a", E1, "en"), ("b", E1_SHOUTED, "en"), ("c", E1_EDITED, "en"), ("d", E3, "en")
    )
    result = grouper().group(frame)
    assert sorted(result.group_sizes()) == [1, 3]
    assert result.edges.near_candidates >= result.edges.near == 1


def test_a_group_above_the_warning_size_is_logged_with_counts_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = messages(*[(f"m{i}", E1, "en") for i in range(4)], ("x", E3, "en"))
    with caplog.at_level(logging.WARNING):
        grouper(max_group_size_warning=3).group(frame)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "4" in warnings[0].getMessage()
    assert E1 not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        grouper(max_group_size_warning=4).group(frame)
    assert not caplog.records
