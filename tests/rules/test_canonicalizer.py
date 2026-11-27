import datetime
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from banhammer.rules.canonicalizer import CanonicalizationResult, Canonicalizer
from banhammer.rules.constants import ClusterResolution, PairOutcome
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import CanonicalizationError, ReviewDecisionError
from banhammer.rules.geometry import cluster_diameter
from banhammer.rules.records import ScoredPair
from banhammer.rules.review import EntailmentDecision, ReviewDecisions, SplitDecision
from banhammer.rules.settings import RulesSettings
from embedder_double import FakeEmbedder
from rule_corpus_builders import make_ruleset

SETTINGS = RulesSettings.model_validate(
    {
        "rulesets_dir": "in",
        "output_dir": "out",
        "review_dir": "review",
        "neighbours_k": 5,
        "neighbours_min_cosine": 0.5,
        "neighbours_chunk_size": 4,
        "entail_high": 0.8,
        "entail_low": 0.2,
        "diameter_max": 0.5,
        "saturation_step": 1,
        "nli": {"model": "org/nli", "revision": "ab" * 20, "batch_size": 2, "max_length": 64},
    }
)
NO_DECISIONS = ReviewDecisions(entailment=(), splits=())


def at_angle(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


def corpus_of(names: list[str]) -> RuleCorpus:
    return RuleCorpus(
        rulesets=tuple(
            make_ruleset(
                f"s{index}",
                [(name, f"text {name}")],
                harvest_date=datetime.date(2026, 11, 1 + index),
            )
            for index, name in enumerate(names)
        )
    )


def embedder_for(angles: dict[str, float]) -> FakeEmbedder:
    return FakeEmbedder({f"text {name}": at_angle(angle) for name, angle in angles.items()})


def pair(a: str, b: str, outcome: PairOutcome = PairOutcome.EDGE) -> ScoredPair:
    high, low = (0.95, 0.05)
    forward, backward = {
        PairOutcome.EDGE: (high, high),
        PairOutcome.NO_EDGE: (low, low),
        PairOutcome.REVIEW: (high, low),
    }[outcome]
    return ScoredPair(
        rule_a=a, rule_b=b, cosine=0.9, p_entail_ab=forward, p_entail_ba=backward, outcome=outcome
    )


def run(
    angles: dict[str, float],
    pairs: list[ScoredPair],
    decisions: ReviewDecisions = NO_DECISIONS,
    settings: RulesSettings = SETTINGS,
) -> tuple[CanonicalizationResult, RuleCorpus]:
    corpus = corpus_of(list(angles))
    return Canonicalizer(embedder_for(angles), settings).canonicalize(
        corpus, pairs, decisions
    ), corpus


def members(result: CanonicalizationResult) -> list[tuple[str, ...]]:
    return [cluster.rule_ids for cluster in result.clusters]


def test_edges_merge_rules_and_every_rule_lands_in_one_cluster() -> None:
    result, _ = run({"a": 0, "b": 5, "c": 90}, [pair("a", "b")])
    assert members(result) == [("a", "b"), ("c",)]
    assert [cluster.cluster_id for cluster in result.clusters] == ["c-a", "c-c"]
    assert result.is_final


def test_rules_chain_through_edges_into_one_cluster() -> None:
    result, _ = run({"a": 0, "b": 5, "c": 10}, [pair("a", "b"), pair("b", "c")])
    assert members(result) == [("a", "b", "c")]


def test_pairs_without_an_edge_do_not_merge() -> None:
    result, _ = run({"a": 0, "b": 5}, [pair("a", "b", PairOutcome.NO_EDGE)])
    assert members(result) == [("a",), ("b",)]


def test_an_accepted_review_pair_becomes_an_edge_and_a_rejected_one_does_not() -> None:
    queued = [pair("a", "b", PairOutcome.REVIEW), pair("a", "c", PairOutcome.REVIEW)]
    decisions = ReviewDecisions(
        entailment=(
            EntailmentDecision(rule_a="a", rule_b="b", accept=True),
            EntailmentDecision(rule_a="a", rule_b="c", accept=False),
        ),
        splits=(),
    )
    result, _ = run({"a": 0, "b": 5, "c": 10}, queued, decisions)
    assert members(result) == [("a", "b"), ("c",)]
    assert result.pending_pairs == ()


def test_an_undecided_review_pair_leaves_canonicalization_pending() -> None:
    queued = pair("a", "b", PairOutcome.REVIEW)
    result, _ = run({"a": 0, "b": 5}, [queued])
    assert result.pending_pairs == (queued,)
    assert not result.is_final
    assert (result.clusters, result.pending_clusters, result.saturation) == ((), (), ())


def test_a_decision_for_a_pair_that_was_not_queued_is_rejected() -> None:
    decisions = ReviewDecisions(
        entailment=(EntailmentDecision(rule_a="a", rule_b="b", accept=True),), splits=()
    )
    with pytest.raises(ReviewDecisionError, match=r"a.*b"):
        run({"a": 0, "b": 5}, [pair("a", "b", PairOutcome.EDGE)], decisions)
    with pytest.raises(ReviewDecisionError, match=r"a.*b"):
        run({"a": 0, "b": 5}, [], decisions)


def test_pairs_naming_unknown_rules_are_rejected() -> None:
    with pytest.raises(CanonicalizationError, match="ghost"):
        run({"a": 0}, [pair("a", "ghost")])


def test_a_chain_whose_ends_are_far_apart_goes_to_split_review() -> None:
    result, _ = run({"a": 0, "b": 45, "c": 90}, [pair("a", "b"), pair("b", "c")])
    assert not result.is_final
    (flagged,) = result.pending_clusters
    assert flagged.rule_ids == ("a", "b", "c")
    assert flagged.resolution is ClusterResolution.PENDING
    assert flagged.diameter == pytest.approx(1.0)
    assert result.clusters == ()


def test_a_cluster_at_exactly_the_diameter_limit_passes() -> None:
    angles = {"a": 0.0, "b": 30.0}
    exact = cluster_diameter(embedder_for(angles).encode(["text a", "text b"]))
    result, _ = run(
        angles, [pair("a", "b")], settings=SETTINGS.model_copy(update={"diameter_max": exact})
    )
    (cluster,) = result.clusters
    assert cluster.resolution is ClusterResolution.AUTO
    assert cluster.diameter == pytest.approx(exact, abs=1e-6)
    tighter = SETTINGS.model_copy(update={"diameter_max": exact - 1e-6})
    flagged, _ = run(angles, [pair("a", "b")], settings=tighter)
    assert len(flagged.pending_clusters) == 1


def test_keeping_a_flagged_cluster_whole_records_the_decision() -> None:
    decisions = ReviewDecisions(
        entailment=(), splits=(SplitDecision(cluster_id="c-a", groups=(("a", "b", "c"),)),)
    )
    result, _ = run({"a": 0, "b": 45, "c": 90}, [pair("a", "b"), pair("b", "c")], decisions)
    (cluster,) = result.clusters
    assert cluster.resolution is ClusterResolution.KEPT
    assert cluster.diameter == pytest.approx(1.0)
    assert result.is_final


def test_splitting_a_flagged_cluster_makes_one_cluster_per_group() -> None:
    decisions = ReviewDecisions(
        entailment=(),
        splits=(SplitDecision(cluster_id="c-a", groups=(("c",), ("a", "b"))),),
    )
    result, corpus = run({"a": 0, "b": 45, "c": 90}, [pair("a", "b"), pair("b", "c")], decisions)
    assert members(result) == [("a", "b"), ("c",)]
    assert [c.resolution for c in result.clusters] == [ClusterResolution.SPLIT] * 2
    assert [c.cluster_id for c in result.clusters] == ["c-a", "c-c"]
    assert result.clusters[0].diameter == pytest.approx(1 - math.cos(math.radians(45)))
    assert result.saturation[-1].cluster_count == 2
    assert result.saturation[-1].rule_count == len(corpus.rule_ids)


def test_clusters_stay_ordered_by_id_when_a_split_interleaves_with_others() -> None:
    decisions = ReviewDecisions(
        entailment=(),
        splits=(SplitDecision(cluster_id="c-a", groups=(("a", "b"), ("c",))),),
    )
    angles = {"a": 0, "b": 45, "c": 90, "bb": 200}
    result, _ = run(angles, [pair("a", "b"), pair("b", "c")], decisions)
    assert [cluster.cluster_id for cluster in result.clusters] == ["c-a", "c-bb", "c-c"]


def test_a_split_decision_must_partition_exactly_the_cluster_members() -> None:
    for groups in ((("a", "b"),), (("a", "b"), ("c", "d")), (("a",), ("b",))):
        decisions = ReviewDecisions(
            entailment=(), splits=(SplitDecision(cluster_id="c-a", groups=groups),)
        )
        with pytest.raises(ReviewDecisionError, match="c-a"):
            run({"a": 0, "b": 45, "c": 90}, [pair("a", "b"), pair("b", "c")], decisions)


def test_a_split_decision_for_a_cluster_that_was_not_flagged_is_rejected() -> None:
    decisions = ReviewDecisions(
        entailment=(), splits=(SplitDecision(cluster_id="c-a", groups=(("a", "b"),)),)
    )
    with pytest.raises(ReviewDecisionError, match="c-a"):
        run({"a": 0, "b": 5}, [pair("a", "b")], decisions)


def test_the_saturation_curve_follows_harvest_order_and_the_configured_step() -> None:
    result, _ = run({"a": 0, "b": 5, "c": 90}, [pair("a", "b")])
    assert [(p.rulesets_added, p.cluster_count) for p in result.saturation] == [
        (1, 1),
        (2, 1),
        (3, 2),
    ]
    stepped = SETTINGS.model_copy(update={"saturation_step": 2})
    result, _ = run({"a": 0, "b": 5, "c": 90}, [pair("a", "b")], settings=stepped)
    assert [p.rulesets_added for p in result.saturation] == [2, 3]


def test_item_texts_are_embedded_once_in_corpus_order() -> None:
    corpus = corpus_of(["a", "b"])
    embedder = embedder_for({"a": 0, "b": 5})
    Canonicalizer(embedder, SETTINGS).canonicalize(corpus, [], NO_DECISIONS)
    assert embedder.calls == [("text a", "text b")]


edge_lists = st.lists(
    st.tuples(st.integers(0, 7), st.integers(0, 7)).filter(lambda e: e[0] < e[1]), max_size=12
)


def components(count: int, edges: list[tuple[int, int]]) -> set[frozenset[int]]:
    reach = {node: {node} for node in range(count)}
    for first, second in edges:
        merged = reach[first] | reach[second]
        for node in merged:
            reach[node] = merged
    return {frozenset(group) for group in reach.values()}


@given(edge_lists, st.data())
def test_clusters_are_the_connected_components_whatever_the_order_of_the_edges(
    edges: list[tuple[int, int]], data: st.DataObject
) -> None:
    names = [f"r{index}" for index in range(8)]
    angles = {name: float(index) for index, name in enumerate(names)}
    shuffled = data.draw(st.permutations(edges))
    first, _ = run(angles, [pair(names[a], names[b]) for a, b in edges])
    second, _ = run(angles, [pair(names[a], names[b]) for a, b in shuffled])
    assert first == second
    found = {
        frozenset(int(rule_id[1:]) for rule_id in cluster.rule_ids) for cluster in first.clusters
    }
    assert found == components(8, edges)
