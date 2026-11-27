"""The saturation curve: how many canonical clusters the rulesets reach as harvesting goes on.

Harvesting stops at saturation, when the marginal rate of new canonical clusters falls below
5% per additional 10 rulesets (proposal 4.2). The curve adds rulesets in harvest order and
reads the cluster count of an incremental union-find after each step (D15).
"""

from collections import Counter
from collections.abc import Sequence

from banhammer.core.union_find import UnionFind
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import CanonicalizationError
from banhammer.rules.records import Cluster, SaturationPoint


class SaturationCurveBuilder:
    """Builds the curve from the final clusters, so split decisions are reflected in it."""

    def __init__(self, step: int) -> None:
        """Record one point every `step` rulesets, and one at the last ruleset.

        Raises:
            CanonicalizationError: `step` is not positive.
        """
        if step < 1:
            message = f"saturation step must be positive, got {step}"
            raise CanonicalizationError(message)
        self._step = step

    def build(self, corpus: RuleCorpus, clusters: Sequence[Cluster]) -> tuple[SaturationPoint, ...]:
        """The curve of `corpus` under `clusters`, with the rulesets added in harvest order.

        A union-find over rule ids gains the rules of each ruleset, joined to the first rule of
        the same cluster already present, so its group count is the number of clusters reached.

        Raises:
            CanonicalizationError: `clusters` do not partition exactly the rules of `corpus`.
        """
        cluster_of = self._cluster_of(corpus, clusters)
        rules_of = {ruleset.ruleset_id: ruleset.rules for ruleset in corpus.rulesets}
        union_find: UnionFind[str] = UnionFind()
        anchors: dict[str, str] = {}
        points: list[SaturationPoint] = []
        previous_count = 0
        for added, ruleset_id in enumerate(corpus.harvest_order, start=1):
            for rule in rules_of[ruleset_id]:
                union_find.add(rule.rule_id)
                anchor = anchors.setdefault(cluster_of[rule.rule_id], rule.rule_id)
                union_find.union(anchor, rule.rule_id)
            if added % self._step == 0 or added == len(corpus.harvest_order):
                count = union_find.group_count
                points.append(
                    SaturationPoint(
                        rulesets_added=added,
                        rule_count=len(union_find),
                        cluster_count=count,
                        new_clusters=count - previous_count,
                    )
                )
                previous_count = count
        return tuple(points)

    @staticmethod
    def _cluster_of(corpus: RuleCorpus, clusters: Sequence[Cluster]) -> dict[str, str]:
        placed = Counter(rule_id for cluster in clusters for rule_id in cluster.rule_ids)
        missing = sorted(set(corpus.rule_ids) - set(placed))
        unknown = sorted(set(placed) - set(corpus.rule_ids))
        repeated = sorted(rule_id for rule_id, count in placed.items() if count > 1)
        if missing or unknown or repeated:
            message = (
                f"clusters must partition the corpus: missing {missing}, unknown {unknown}, "
                f"in several clusters {repeated}"
            )
            raise CanonicalizationError(message)
        return {rule_id: c.cluster_id for c in clusters for rule_id in c.rule_ids}
