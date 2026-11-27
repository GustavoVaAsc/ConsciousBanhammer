import datetime
from collections.abc import Callable
from typing import Any

import pandera.errors
import polars as pl
import pytest

from banhammer.rules.schemas import (
    CandidatePairsSchema,
    CanonicalClustersSchema,
    RulesetsSchema,
    RulesSchema,
    SaturationCurveSchema,
)


def rules_frame(**overrides: list[Any]) -> pl.DataFrame:
    columns: dict[str, list[Any]] = {
        "rule_id": ["a-1", "a-2"],
        "ruleset_id": ["a", "a"],
        "text": ["No insults.", "Be kind."],
        "scope": [None, "Comments"],
        "exceptions": [None, None],
        "sanction_hint": ["ban", None],
        "rule_type": ["conduct", "meta"],
        "discretion_flag": [False, True],
    }
    return pl.DataFrame(
        columns | overrides,
        schema_overrides={"scope": pl.String, "exceptions": pl.String, "sanction_hint": pl.String},
    )


def rulesets_frame(**overrides: list[Any]) -> pl.DataFrame:
    columns: dict[str, list[Any]] = {
        "ruleset_id": ["a", "b"],
        "platform": ["reddit", "discord"],
        "language": ["en", "es"],
        "source_url": ["https://example.org/a", "http://example.org/b"],
        "harvest_date": [datetime.date(2026, 11, 1), datetime.date(2026, 11, 2)],
        "terms_note": ["Public.", "Public."],
        "rule_count": [2, 1],
    }
    return pl.DataFrame(columns | overrides)


def candidates_frame(**overrides: list[Any]) -> pl.DataFrame:
    columns: dict[str, list[Any]] = {
        "rule_a": ["a-1", "a-1"],
        "rule_b": ["b-1", "c-1"],
        "cosine": [0.9, 0.6],
        "p_entail_ab": [0.95, 0.5],
        "p_entail_ba": [0.9, 0.1],
        "outcome": ["edge", "review"],
    }
    return pl.DataFrame(columns | overrides)


def clusters_frame(**overrides: list[Any]) -> pl.DataFrame:
    columns: dict[str, list[Any]] = {
        "cluster_id": ["c-a-1", "c-a-1", "c-b-9"],
        "rule_id": ["a-1", "b-1", "b-9"],
        "ruleset_id": ["a", "b", "b"],
        "cluster_size": [2, 2, 1],
        "diameter": [0.3, 0.3, 0.0],
        "resolution": ["auto", "auto", "auto"],
    }
    return pl.DataFrame(columns | overrides)


def curve_frame(**overrides: list[Any]) -> pl.DataFrame:
    columns: dict[str, list[Any]] = {
        "rulesets_added": [10, 20, 25],
        "rule_count": [100, 200, 240],
        "cluster_count": [60, 80, 85],
        "new_clusters": [60, 20, 5],
    }
    return pl.DataFrame(columns | overrides)


CASES: list[tuple[Any, Callable[..., pl.DataFrame]]] = [
    (RulesSchema, rules_frame),
    (RulesetsSchema, rulesets_frame),
    (CandidatePairsSchema, candidates_frame),
    (CanonicalClustersSchema, clusters_frame),
    (SaturationCurveSchema, curve_frame),
]


@pytest.mark.parametrize(("schema", "build"), CASES)
def test_a_valid_frame_passes(schema: Any, build: Callable[..., pl.DataFrame]) -> None:
    assert schema.validate(build(), lazy=True).shape == build().shape


@pytest.mark.parametrize(("schema", "build"), CASES)
def test_an_extra_column_is_rejected(schema: Any, build: Callable[..., pl.DataFrame]) -> None:
    frame = build().with_columns(pl.lit(1).alias("unexpected"))
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(frame, lazy=True)


@pytest.mark.parametrize(("schema", "build"), CASES)
def test_a_missing_column_is_rejected(schema: Any, build: Callable[..., pl.DataFrame]) -> None:
    frame = build().drop(build().columns[-1])
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(frame, lazy=True)


def test_the_rules_columns_are_the_eight_fields_in_order() -> None:
    assert list(RulesSchema.to_schema().columns) == [
        "rule_id",
        "ruleset_id",
        "text",
        "scope",
        "exceptions",
        "sanction_hint",
        "rule_type",
        "discretion_flag",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        {"rule_id": ["a-1", "a-1"]},
        {"rule_id": ["A 1", "a-2"]},
        {"text": ["", "Be kind."]},
        {"scope": ["", "Comments"]},
        {"rule_type": ["conduct", "spam"]},
        {"discretion_flag": [None, True]},
        {"text": [None, "Be kind."]},
    ],
)
def test_invalid_rules_are_rejected(bad: dict[str, list[Any]]) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        RulesSchema.validate(rules_frame(**bad), lazy=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"ruleset_id": ["a", "a"]},
        {"language": ["en", "fr"]},
        {"platform": ["Reddit", "discord"]},
        {"source_url": ["example.org", "http://example.org/b"]},
        {"rule_count": [0, 1]},
        {"terms_note": ["", "Public."]},
    ],
)
def test_invalid_rulesets_are_rejected(bad: dict[str, list[Any]]) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        RulesetsSchema.validate(rulesets_frame(**bad), lazy=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"rule_a": ["b-1", "a-1"], "rule_b": ["a-1", "c-1"]},
        {"rule_b": ["b-1", "b-1"]},
        {"cosine": [1.5, 0.6]},
        {"p_entail_ab": [-0.1, 0.5]},
        {"p_entail_ba": [0.9, 1.1]},
        {"outcome": ["edge", "maybe"]},
        {"rule_a": ["a-1", "a-1"], "rule_b": ["a-1", "c-1"]},
    ],
)
def test_invalid_candidate_pairs_are_rejected(bad: dict[str, list[Any]]) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        CandidatePairsSchema.validate(candidates_frame(**bad), lazy=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"rule_id": ["a-1", "a-1", "b-9"]},
        {"cluster_size": [3, 3, 1]},
        {"cluster_size": [2, 2, 2]},
        {"diameter": [0.3, 0.4, 0.0]},
        {"diameter": [0.3, 0.3, 2.5]},
        {"resolution": ["auto", "kept", "auto"]},
        {"resolution": ["auto", "auto", "pending"]},
        {"cluster_id": ["c-b-1", "c-b-1", "c-b-9"]},
        {"cluster_id": ["c-a-1", "c-b-1", "c-b-9"]},
    ],
)
def test_invalid_clusters_are_rejected(bad: dict[str, list[Any]]) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        CanonicalClustersSchema.validate(clusters_frame(**bad), lazy=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"rulesets_added": [10, 10, 25]},
        {"rulesets_added": [20, 10, 25]},
        {"new_clusters": [60, 20, 6]},
        {"cluster_count": [60, 80, 70], "new_clusters": [60, 20, -10]},
        {"rule_count": [0, 200, 240]},
    ],
)
def test_invalid_saturation_curves_are_rejected(bad: dict[str, list[Any]]) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        SaturationCurveSchema.validate(curve_frame(**bad), lazy=True)
