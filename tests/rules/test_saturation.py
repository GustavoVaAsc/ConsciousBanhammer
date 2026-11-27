import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from banhammer.rules.constants import CLUSTER_ID_PREFIX, ClusterResolution
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import CanonicalizationError
from banhammer.rules.records import Cluster, SaturationPoint
from banhammer.rules.saturation import SaturationCurveBuilder
from rule_corpus_builders import make_ruleset


def cluster(*rule_ids: str) -> Cluster:
    members = tuple(sorted(rule_ids))
    return Cluster(
        cluster_id=f"{CLUSTER_ID_PREFIX}{members[0]}",
        rule_ids=members,
        diameter=0.0,
        resolution=ClusterResolution.AUTO,
    )


def harvest_corpus() -> RuleCorpus:
    return RuleCorpus(
        rulesets=(
            make_ruleset("r3", [("r3-a", "c")], harvest_date=datetime.date(2026, 11, 3)),
            make_ruleset(
                "r1", [("r1-a", "x"), ("r1-b", "y")], harvest_date=datetime.date(2026, 11, 1)
            ),
            make_ruleset(
                "r2", [("r2-a", "x2"), ("r2-b", "z")], harvest_date=datetime.date(2026, 11, 2)
            ),
        )
    )


CLUSTERS = (cluster("r1-a", "r2-a"), cluster("r1-b"), cluster("r2-b", "r3-a"))


def points(step: int) -> list[tuple[int, int, int, int]]:
    curve = SaturationCurveBuilder(step).build(harvest_corpus(), CLUSTERS)
    return [(p.rulesets_added, p.rule_count, p.cluster_count, p.new_clusters) for p in curve]


def test_a_point_after_every_ruleset_counts_the_clusters_reached_so_far() -> None:
    assert points(1) == [(1, 2, 2, 2), (2, 4, 3, 1), (3, 5, 3, 0)]


def test_a_larger_step_samples_the_same_curve_and_ends_at_the_last_ruleset() -> None:
    assert points(2) == [(2, 4, 3, 3), (3, 5, 3, 0)]


def test_a_step_beyond_the_corpus_gives_one_point() -> None:
    assert points(10) == [(3, 5, 3, 3)]


def test_rulesets_are_added_in_harvest_order_not_id_order() -> None:
    reordered = RuleCorpus(
        rulesets=(
            make_ruleset(
                "r1", [("r1-a", "x"), ("r1-b", "y")], harvest_date=datetime.date(2026, 11, 3)
            ),
            make_ruleset(
                "r2", [("r2-a", "x2"), ("r2-b", "z")], harvest_date=datetime.date(2026, 11, 2)
            ),
            make_ruleset("r3", [("r3-a", "c")], harvest_date=datetime.date(2026, 11, 1)),
        )
    )
    curve = SaturationCurveBuilder(1).build(reordered, CLUSTERS)
    assert [p.cluster_count for p in curve] == [1, 2, 3]


def test_a_cluster_spanning_rulesets_is_counted_once() -> None:
    curve = SaturationCurveBuilder(1).build(
        harvest_corpus(), (cluster("r1-a", "r2-a", "r1-b", "r2-b", "r3-a"),)
    )
    assert [p.cluster_count for p in curve] == [1, 1, 1]


def test_clusters_that_do_not_cover_the_corpus_are_rejected() -> None:
    with pytest.raises(CanonicalizationError, match="r3-a"):
        SaturationCurveBuilder(1).build(harvest_corpus(), CLUSTERS[:2])


def test_clusters_naming_unknown_rules_are_rejected() -> None:
    with pytest.raises(CanonicalizationError, match="ghost"):
        SaturationCurveBuilder(1).build(harvest_corpus(), (*CLUSTERS, cluster("ghost")))


def test_a_rule_in_two_clusters_is_rejected() -> None:
    with pytest.raises(CanonicalizationError, match="r1-a"):
        SaturationCurveBuilder(1).build(harvest_corpus(), (*CLUSTERS, cluster("r1-a", "ghost")))


def test_the_step_must_be_positive() -> None:
    with pytest.raises(CanonicalizationError, match="step"):
        SaturationCurveBuilder(0)


@given(
    st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=12),
    st.integers(min_value=1, max_value=5),
)
def test_the_curve_is_monotone_and_ends_at_the_final_cluster_count(
    assignment: list[int], step: int
) -> None:
    rule_ids = [f"r{index:02d}-a" for index in range(len(assignment))]
    corpus = RuleCorpus(
        rulesets=tuple(
            make_ruleset(
                f"r{index:02d}", [(rule_id, "t")], harvest_date=datetime.date(2026, 1, 1 + index)
            )
            for index, rule_id in enumerate(rule_ids)
        )
    )
    members: dict[int, list[str]] = {}
    for rule_id, label in zip(rule_ids, assignment, strict=True):
        members.setdefault(label, []).append(rule_id)
    curve: tuple[SaturationPoint, ...] = SaturationCurveBuilder(step).build(
        corpus, [cluster(*ids) for ids in members.values()]
    )
    counts = [point.cluster_count for point in curve]
    assert counts == sorted(counts)
    assert counts[-1] == len(members)
    assert sum(point.new_clusters for point in curve) == len(members)
    assert curve[-1].rulesets_added == len(rule_ids)
