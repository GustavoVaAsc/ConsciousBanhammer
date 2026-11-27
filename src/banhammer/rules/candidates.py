"""Candidate rule pairs and their mutual-entailment outcome (D15).

Items are embedded, each item's nearest neighbours above a cosine floor become candidates, and
an entailment model scores both directions of every candidate.
"""

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from banhammer.embeddings import Embedder, ExactNeighbourSearch
from banhammer.rules.constants import COSINE_MAX, COSINE_MIN, SCORE_DECIMALS, PairOutcome
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.entailment import EntailmentScorer
from banhammer.rules.errors import EntailmentError
from banhammer.rules.records import ScoredPair
from banhammer.rules.settings import RulesSettings

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Finds candidate pairs across rulesets and classifies them by mutual entailment.

    Pairs inside one ruleset are not candidates: the community itself kept those items apart,
    and "No insults" with "No insults toward streamers" in one ruleset would only fill the
    review queue. The search runs ruleset by ruleset against the items of all other rulesets, so
    siblings never use up the `k` places of a cross-ruleset match. A pair is an edge when
    entailment is at least `entail_high` both ways, no edge when it is at most `entail_low` both
    ways, and a review pair otherwise. The outcome is decided on the float32 probabilities
    against float32 thresholds; the stored scores are rounded afterwards.
    """

    def __init__(
        self,
        embedder: Embedder,
        search: ExactNeighbourSearch,
        scorer: EntailmentScorer,
        settings: RulesSettings,
    ) -> None:
        """Take the collaborators; `search` carries `k`, the cosine floor and the device."""
        self._embedder = embedder
        self._search = search
        self._scorer = scorer
        self._settings = settings

    def generate(self, corpus: RuleCorpus) -> tuple[ScoredPair, ...]:
        """Scored candidate pairs, once per unordered pair, sorted by `(rule_a, rule_b)`.

        Raises:
            EntailmentError: The scorer returns the wrong number of values or values outside
                `[0, 1]`.
        """
        cosines = self._candidate_cosines(corpus)
        texts = dict(zip(corpus.rule_ids, corpus.item_texts, strict=True))
        keys = sorted(cosines)
        premises = [texts[first] for first, _ in keys]
        hypotheses = [texts[second] for _, second in keys]
        forward = self._probabilities(premises, hypotheses)
        backward = self._probabilities(hypotheses, premises)
        pairs = tuple(
            self._scored_pair(key, cosines[key], p_forward, p_backward)
            for key, p_forward, p_backward in zip(keys, forward, backward, strict=True)
        )
        logger.info("scored %d candidate pairs", len(pairs))
        return pairs

    def _candidate_cosines(self, corpus: RuleCorpus) -> dict[tuple[str, str], float]:
        embeddings = self._embedder.encode(corpus.item_texts)
        rules = corpus.rules
        ruleset_ids = np.array([rule.ruleset_id for rule in rules])
        cosines: dict[tuple[str, str], float] = {}
        for ruleset in corpus.rulesets:
            own = np.flatnonzero(ruleset_ids == ruleset.ruleset_id)
            others = np.flatnonzero(ruleset_ids != ruleset.ruleset_id)
            if not others.size:
                continue
            found = self._search.between(embeddings[own], embeddings[others])
            for query, neighbour, score in zip(
                own[found.query_indices].tolist(),
                others[found.corpus_indices].tolist(),
                found.scores.tolist(),
                strict=True,
            ):
                first, second = sorted((rules[query].rule_id, rules[neighbour].rule_id))
                cosines[first, second] = max(score, cosines.get((first, second), COSINE_MIN))
        return cosines

    def _probabilities(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> NDArray[np.float32]:
        if not premises:
            return np.empty(0, dtype=np.float32)
        probabilities = self._scorer.entailment_probabilities(premises, hypotheses)
        if probabilities.shape != (len(premises),):
            message = f"expected {len(premises)} probabilities, got shape {probabilities.shape}"
            raise EntailmentError(message)
        if not ((probabilities >= 0) & (probabilities <= 1)).all():
            message = "entailment probabilities outside [0, 1]"
            raise EntailmentError(message)
        return np.asarray(probabilities, dtype=np.float32)

    def _scored_pair(
        self, key: tuple[str, str], cosine: float, forward: np.float32, backward: np.float32
    ) -> ScoredPair:
        return ScoredPair(
            rule_a=key[0],
            rule_b=key[1],
            cosine=round(min(cosine, COSINE_MAX), SCORE_DECIMALS),
            p_entail_ab=round(float(forward), SCORE_DECIMALS),
            p_entail_ba=round(float(backward), SCORE_DECIMALS),
            outcome=self._outcome(forward, backward),
        )

    def _outcome(self, forward: np.float32, backward: np.float32) -> PairOutcome:
        high = np.float32(self._settings.entail_high)
        low = np.float32(self._settings.entail_low)
        if min(forward, backward) >= high:
            return PairOutcome.EDGE
        if max(forward, backward) <= low:
            return PairOutcome.NO_EDGE
        return PairOutcome.REVIEW
