import datetime

import polars as pl
import pytest

from banhammer.core.constants import Language
from banhammer.rules.constants import ClusterResolution, PairOutcome
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import SchemaViolationError
from banhammer.rules.frames import (
    candidate_pairs_frame,
    candidate_pairs_from_frame,
    clusters_frame,
    corpus_from_frames,
    rules_frame,
    rulesets_frame,
    saturation_frame,
)
from banhammer.rules.records import Cluster, SaturationPoint, ScoredPair
from banhammer.rules.schemas import (
    CandidatePairsSchema,
    CanonicalClustersSchema,
    RulesetsSchema,
    RulesSchema,
    SaturationCurveSchema,
)
from rule_corpus_builders import make_rule, make_ruleset


def corpus() -> RuleCorpus:
    rich = make_ruleset("b", [("b-1", "No insultos.")], language=Language.ES)
    rich = rich.model_copy(
        update={
            "rules": (
                make_rule(
                    "b-1",
                    "b",
                    "No insultos.",
                    scope="Comentarios",
                    sanction_hint="ban",
                    discretion_flag=True,
                    rule_type="harm",
                ),
            )
        }
    )
    plain = make_ruleset(
        "a",
        [("a-1", "No insults."), ("a-2", "Be kind.")],
        harvest_date=datetime.date(2026, 11, 3),
        platform="discord",
    )
    return RuleCorpus(rulesets=(rich, plain))


def test_the_rules_frame_follows_the_schema_and_corpus_order() -> None:
    frame = rules_frame(corpus())
    assert RulesSchema.validate(frame, lazy=True).shape == (3, 8)
    assert frame["rule_id"].to_list() == ["a-1", "a-2", "b-1"]
    assert frame["scope"].to_list() == [None, None, "Comentarios"]
    assert frame["sanction_hint"].to_list() == [None, None, "ban"]
    assert frame["rule_type"].to_list() == ["conduct", "conduct", "harm"]
    assert frame["discretion_flag"].to_list() == [False, False, True]


def test_the_rulesets_frame_carries_stratification_attributes_and_provenance() -> None:
    frame = rulesets_frame(corpus())
    assert RulesetsSchema.validate(frame, lazy=True).shape == (2, 7)
    assert frame["ruleset_id"].to_list() == ["a", "b"]
    assert frame["platform"].to_list() == ["discord", "reddit"]
    assert frame["language"].to_list() == ["en", "es"]
    assert frame["rule_count"].to_list() == [2, 1]
    assert frame["harvest_date"].to_list() == [
        datetime.date(2026, 11, 3),
        datetime.date(2026, 11, 1),
    ]


def test_the_corpus_survives_a_round_trip_through_its_frames() -> None:
    original = corpus()
    assert corpus_from_frames(rules_frame(original), rulesets_frame(original)) == original


def test_a_rule_of_an_unknown_ruleset_is_reported() -> None:
    original = corpus()
    with pytest.raises(SchemaViolationError, match="ruleset"):
        corpus_from_frames(rules_frame(original), rulesets_frame(original).head(1))


def test_a_wrong_rule_count_is_reported() -> None:
    original = corpus()
    wrong = rulesets_frame(original).with_columns(pl.col("rule_count") + 1)
    with pytest.raises(SchemaViolationError, match="rule_count"):
        corpus_from_frames(rules_frame(original), wrong)


def test_candidate_pairs_round_trip_and_follow_the_schema() -> None:
    pairs = (
        ScoredPair(
            rule_a="a-1",
            rule_b="b-1",
            cosine=0.9,
            p_entail_ab=0.95,
            p_entail_ba=0.9,
            outcome=PairOutcome.EDGE,
        ),
        ScoredPair(
            rule_a="a-2",
            rule_b="b-1",
            cosine=0.6,
            p_entail_ab=0.5,
            p_entail_ba=0.1,
            outcome=PairOutcome.REVIEW,
        ),
    )
    frame = candidate_pairs_frame(pairs)
    assert CandidatePairsSchema.validate(frame, lazy=True).shape == (2, 6)
    assert frame["outcome"].to_list() == ["edge", "review"]
    assert candidate_pairs_from_frame(frame) == pairs


def test_no_candidate_pairs_give_a_valid_empty_frame() -> None:
    frame = candidate_pairs_frame(())
    assert CandidatePairsSchema.validate(frame, lazy=True).shape == (0, 6)
    assert candidate_pairs_from_frame(frame) == ()


def test_the_clusters_frame_has_one_row_per_member_with_the_cluster_attributes() -> None:
    clusters = (
        Cluster(
            cluster_id="c-a-1",
            rule_ids=("a-1", "b-1"),
            diameter=0.3,
            resolution=ClusterResolution.KEPT,
        ),
        Cluster(
            cluster_id="c-a-2", rule_ids=("a-2",), diameter=0.0, resolution=ClusterResolution.AUTO
        ),
    )
    frame = clusters_frame(clusters, corpus())
    assert CanonicalClustersSchema.validate(frame, lazy=True).shape == (3, 6)
    assert frame.rows() == [
        ("c-a-1", "a-1", "a", 2, pytest.approx(0.3), "kept"),
        ("c-a-1", "b-1", "b", 2, pytest.approx(0.3), "kept"),
        ("c-a-2", "a-2", "a", 1, 0.0, "auto"),
    ]


def test_the_saturation_frame_follows_the_schema() -> None:
    points = (
        SaturationPoint(rulesets_added=1, rule_count=2, cluster_count=2, new_clusters=2),
        SaturationPoint(rulesets_added=2, rule_count=3, cluster_count=2, new_clusters=0),
    )
    frame = saturation_frame(points)
    assert SaturationCurveSchema.validate(frame, lazy=True).shape == (2, 4)
    assert frame["cluster_count"].to_list() == [2, 2]
