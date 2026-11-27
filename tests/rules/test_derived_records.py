from typing import Any

import pytest
from pydantic import ValidationError

from banhammer.rules.constants import ClusterResolution, PairOutcome
from banhammer.rules.records import Cluster, SaturationPoint, ScoredPair


def pair_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rule_a": "a-1",
        "rule_b": "b-1",
        "cosine": 0.9,
        "p_entail_ab": 0.95,
        "p_entail_ba": 0.9,
        "outcome": "edge",
    }
    return fields | overrides


def cluster_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "cluster_id": "c-a-1",
        "rule_ids": ("a-1", "b-1"),
        "diameter": 0.3,
        "resolution": "auto",
    }
    return fields | overrides


def test_a_scored_pair_keeps_its_scores_and_outcome() -> None:
    pair = ScoredPair.model_validate(pair_fields())
    assert pair.outcome is PairOutcome.EDGE
    assert (pair.rule_a, pair.rule_b) == ("a-1", "b-1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rule_a", "b-2"),
        ("rule_b", "a-1"),
        ("cosine", 1.2),
        ("cosine", -1.2),
        ("p_entail_ab", 1.1),
        ("p_entail_ba", -0.1),
        ("outcome", "maybe"),
    ],
)
def test_invalid_scored_pairs_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        ScoredPair.model_validate(pair_fields(**{field: value}))


def test_a_cluster_keeps_its_members_and_resolution() -> None:
    cluster = Cluster.model_validate(cluster_fields())
    assert cluster.rule_ids == ("a-1", "b-1")
    assert cluster.resolution is ClusterResolution.AUTO


@pytest.mark.parametrize(
    "bad",
    [
        {"rule_ids": ()},
        {"rule_ids": ("b-1", "a-1")},
        {"rule_ids": ("a-1", "a-1")},
        {"cluster_id": "c-b-1"},
        {"cluster_id": "a-1"},
        {"diameter": -0.1},
        {"diameter": 2.1},
        {"resolution": "unreviewed"},
    ],
)
def test_invalid_clusters_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Cluster.model_validate(cluster_fields(**bad))


def test_a_cluster_can_be_pending_a_split_decision() -> None:
    cluster = Cluster.model_validate(cluster_fields(resolution="pending"))
    assert cluster.resolution is ClusterResolution.PENDING


def test_a_saturation_point_requires_sensible_counts() -> None:
    point = SaturationPoint(rulesets_added=10, rule_count=100, cluster_count=60, new_clusters=60)
    assert point.cluster_count == 60
    for field in ("rulesets_added", "rule_count", "cluster_count"):
        fields = {"rulesets_added": 10, "rule_count": 100, "cluster_count": 60, "new_clusters": 0}
        with pytest.raises(ValidationError, match=field):
            SaturationPoint.model_validate(fields | {field: 0})
    with pytest.raises(ValidationError, match="new_clusters"):
        SaturationPoint(rulesets_added=10, rule_count=100, cluster_count=60, new_clusters=-1)
