"""Canonical rule clusters from entailment edges, review decisions and the diameter check (D15).

Edges are the pairs the entailment model accepted plus the review pairs a human accepted. Their
connected components, found with a union-find, are the cluster candidates. A candidate whose
diameter exceeds the limit is chained rather than coherent and needs a split decision.
"""

import logging
from collections.abc import Mapping, Sequence
from operator import attrgetter

import numpy as np
from numpy.typing import NDArray

from banhammer.core.union_find import UnionFind
from banhammer.embeddings import Embedder
from banhammer.rules.constants import (
    SCORE_DECIMALS,
    ClusterResolution,
    PairOutcome,
)
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import CanonicalizationError, ReviewDecisionError
from banhammer.rules.geometry import cluster_diameter
from banhammer.rules.records import (
    Cluster,
    RulesRecord,
    SaturationPoint,
    ScoredPair,
    cluster_id_of,
)
from banhammer.rules.review import EntailmentDecision, ReviewDecisions, SplitDecision
from banhammer.rules.saturation import SaturationCurveBuilder
from banhammer.rules.settings import RulesSettings

logger = logging.getLogger(__name__)


class CanonicalizationResult(RulesRecord):
    """What canonicalization produced, final or waiting on a human.

    The result is final when no review pair lacks a decision and no cluster over the diameter
    limit lacks a split decision. Only a final result carries clusters and a saturation curve:
    review pairs come first, and clusters are only formed once all of them are answered.
    """

    clusters: tuple[Cluster, ...]
    pending_pairs: tuple[ScoredPair, ...]
    pending_clusters: tuple[Cluster, ...]
    saturation: tuple[SaturationPoint, ...]

    @property
    def is_final(self) -> bool:
        """Whether the clusters can be written as the canonical clusters boundary."""
        return not self.pending_pairs and not self.pending_clusters


class Canonicalizer:
    """Turns scored candidate pairs and review decisions into canonical clusters."""

    def __init__(self, embedder: Embedder, settings: RulesSettings) -> None:
        """Embed with `embedder` for the diameter check; `settings` give the limit and the step."""
        self._embedder = embedder
        self._settings = settings

    def canonicalize(
        self, corpus: RuleCorpus, pairs: Sequence[ScoredPair], decisions: ReviewDecisions
    ) -> CanonicalizationResult:
        """Cluster the rules of `corpus` by the edges of `pairs` and the accepted decisions.

        Raises:
            CanonicalizationError: A pair names a rule that is not in `corpus`.
            ReviewDecisionError: A decision answers a pair that was not queued or a cluster
                that was not flagged, or a split decision is not a partition of its cluster.
        """
        self._require_known_rules(corpus, pairs)
        edges, pending_pairs = self._resolve_pairs(pairs, decisions.entailment)
        if pending_pairs:
            logger.info("%d review pairs await a decision", len(pending_pairs))
            return CanonicalizationResult(
                clusters=(), pending_pairs=pending_pairs, pending_clusters=(), saturation=()
            )
        clusters, pending_clusters = self._check_diameters(corpus, edges, decisions.splits)
        if pending_clusters:
            logger.info("%d clusters await a split decision", len(pending_clusters))
            return CanonicalizationResult(
                clusters=(), pending_pairs=(), pending_clusters=pending_clusters, saturation=()
            )
        saturation = SaturationCurveBuilder(self._settings.saturation_step).build(corpus, clusters)
        logger.info("%d canonical clusters from %d rules", len(clusters), len(corpus.rule_ids))
        return CanonicalizationResult(
            clusters=clusters, pending_pairs=(), pending_clusters=(), saturation=saturation
        )

    @staticmethod
    def _require_known_rules(corpus: RuleCorpus, pairs: Sequence[ScoredPair]) -> None:
        named = {rule_id for pair in pairs for rule_id in (pair.rule_a, pair.rule_b)}
        unknown = sorted(named - set(corpus.rule_ids))
        if unknown:
            message = f"candidate pairs name rules that are not in the corpus: {unknown}"
            raise CanonicalizationError(message)

    @staticmethod
    def _resolve_pairs(
        pairs: Sequence[ScoredPair], decisions: Sequence[EntailmentDecision]
    ) -> tuple[list[tuple[str, str]], tuple[ScoredPair, ...]]:
        queued = {(p.rule_a, p.rule_b) for p in pairs if p.outcome is PairOutcome.REVIEW}
        answers = {(d.rule_a, d.rule_b): d.accept for d in decisions}
        stale = sorted(set(answers) - queued)
        if stale:
            message = f"decisions answer pairs that are not in the review queue: {stale}"
            raise ReviewDecisionError(message)
        edges = [(p.rule_a, p.rule_b) for p in pairs if p.outcome is PairOutcome.EDGE]
        edges += [key for key, accept in answers.items() if accept]
        pending = tuple(
            p
            for p in pairs
            if p.outcome is PairOutcome.REVIEW and (p.rule_a, p.rule_b) not in answers
        )
        return edges, pending

    def _check_diameters(
        self, corpus: RuleCorpus, edges: list[tuple[str, str]], splits: Sequence[SplitDecision]
    ) -> tuple[tuple[Cluster, ...], tuple[Cluster, ...]]:
        union_find: UnionFind[str] = UnionFind(corpus.rule_ids)
        union_find.union_all(edges)
        vectors = dict(zip(corpus.rule_ids, self._embedder.encode(corpus.item_texts), strict=True))
        groups = tuple(union_find.groups().values())
        flagged = {
            cluster_id_of(group): group
            for group in groups
            if self._diameter(group, vectors) > self._settings.diameter_max
        }
        answers = self._answers(flagged, splits)
        clusters: list[Cluster] = []
        pending: list[Cluster] = []
        for group in groups:
            cluster_id = cluster_id_of(group)
            if cluster_id not in flagged:
                clusters.append(self._cluster(group, vectors, ClusterResolution.AUTO))
            elif cluster_id in answers:
                clusters.extend(self._resolve(answers[cluster_id], vectors))
            else:
                pending.append(self._cluster(group, vectors, ClusterResolution.PENDING))
        return tuple(sorted(clusters, key=attrgetter("cluster_id"))), tuple(pending)

    @staticmethod
    def _answers(
        flagged: Mapping[str, tuple[str, ...]], splits: Sequence[SplitDecision]
    ) -> dict[str, SplitDecision]:
        stale = sorted({d.cluster_id for d in splits} - set(flagged))
        if stale:
            message = f"split decisions answer clusters that were not flagged: {stale}"
            raise ReviewDecisionError(message)
        for decision in splits:
            placed = sorted(rule_id for group in decision.groups for rule_id in group)
            if placed != list(flagged[decision.cluster_id]):
                message = (
                    f"split decision for {decision.cluster_id} must partition exactly "
                    f"{list(flagged[decision.cluster_id])}, got {placed}"
                )
                raise ReviewDecisionError(message)
        return {decision.cluster_id: decision for decision in splits}

    def _resolve(
        self, decision: SplitDecision, vectors: Mapping[str, NDArray[np.float32]]
    ) -> list[Cluster]:
        resolution = ClusterResolution.SPLIT if decision.splits_cluster else ClusterResolution.KEPT
        groups = sorted(tuple(sorted(group)) for group in decision.groups)
        return [self._cluster(group, vectors, resolution) for group in groups]

    @staticmethod
    def _diameter(group: Sequence[str], vectors: Mapping[str, NDArray[np.float32]]) -> float:
        return cluster_diameter(np.stack([vectors[rule_id] for rule_id in group]))

    def _cluster(
        self,
        group: Sequence[str],
        vectors: Mapping[str, NDArray[np.float32]],
        resolution: ClusterResolution,
    ) -> Cluster:
        return Cluster(
            cluster_id=cluster_id_of(group),
            rule_ids=tuple(group),
            diameter=round(self._diameter(group, vectors), SCORE_DECIMALS),
            resolution=resolution,
        )
