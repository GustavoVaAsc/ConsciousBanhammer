import datetime
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from banhammer.core.constants import Language
from banhammer.embeddings import ExactNeighbourSearch
from banhammer.rules.candidates import CandidateGenerator
from banhammer.rules.constants import PairOutcome
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import EntailmentError
from banhammer.rules.settings import RulesSettings
from embedder_double import FakeEmbedder
from entailment_double import FakeEntailmentScorer
from rule_corpus_builders import make_rule, make_ruleset

CPU = torch.device("cpu")
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
        "saturation_step": 10,
        "nli": {"model": "org/nli", "revision": "ab" * 20, "batch_size": 2, "max_length": 64},
    }
)

INSULTS_EN = "No insults."
INSULTS_ES = "No insultos."
STREAMERS = "No insults toward streamers."
SPAM = "No spam."
OFFTOPIC = "Stay on topic."

CORPUS = RuleCorpus(
    rulesets=(
        make_ruleset("a", [("a-1", INSULTS_EN), ("a-2", SPAM)]),
        make_ruleset(
            "b",
            [("b-1", INSULTS_ES)],
            language=Language.ES,
            harvest_date=datetime.date(2026, 11, 2),
        ),
        make_ruleset("c", [("c-1", STREAMERS), ("c-2", OFFTOPIC)]),
    )
)
VECTORS = {
    INSULTS_EN: [1.0, 0.0, 0.0],
    INSULTS_ES: [0.98, 0.2, 0.0],
    STREAMERS: [0.9, 0.0, 0.44],
    SPAM: [0.6, 0.8, 0.0],
    OFFTOPIC: [0.0, 0.0, 1.0],
}


def generator(
    probabilities: dict[tuple[str, str], float] | None = None,
    *,
    k: int = 5,
    embedder: FakeEmbedder | None = None,
) -> tuple[CandidateGenerator, FakeEntailmentScorer]:
    scorer = FakeEntailmentScorer(probabilities or {})
    search = ExactNeighbourSearch(k=k, chunk_size=4, device=CPU, min_score=0.5)
    return CandidateGenerator(embedder or FakeEmbedder(VECTORS), search, scorer, SETTINGS), scorer


BOTH_WAYS = {
    (INSULTS_EN, INSULTS_ES): 0.95,
    (INSULTS_ES, INSULTS_EN): 0.9,
    (INSULTS_EN, STREAMERS): 0.1,
    (STREAMERS, INSULTS_EN): 0.9,
}


def by_pair(generated: Any) -> dict[tuple[str, str], Any]:
    return {(pair.rule_a, pair.rule_b): pair for pair in generated}


def test_a_cross_lingual_pair_entailing_both_ways_is_an_edge() -> None:
    pairs = by_pair(generator(BOTH_WAYS)[0].generate(CORPUS))
    pair = pairs[("a-1", "b-1")]
    assert pair.outcome is PairOutcome.EDGE
    assert pair.p_entail_ab == pytest.approx(0.95)
    assert pair.p_entail_ba == pytest.approx(0.9)
    assert 0.9 < pair.cosine <= 1.0


def test_a_pair_entailing_one_way_only_goes_to_review() -> None:
    pair = by_pair(generator(BOTH_WAYS)[0].generate(CORPUS))[("a-1", "c-1")]
    assert pair.outcome is PairOutcome.REVIEW
    assert (pair.p_entail_ab, pair.p_entail_ba) == (pytest.approx(0.1), pytest.approx(0.9))


def test_a_pair_entailing_neither_way_is_no_edge() -> None:
    pair = by_pair(generator(BOTH_WAYS)[0].generate(CORPUS))[("a-2", "b-1")]
    assert pair.outcome is PairOutcome.NO_EDGE


def test_probabilities_in_the_middle_band_go_to_review() -> None:
    table = {(INSULTS_EN, INSULTS_ES): 0.9, (INSULTS_ES, INSULTS_EN): 0.5}
    pair = by_pair(generator(table)[0].generate(CORPUS))[("a-1", "b-1")]
    assert pair.outcome is PairOutcome.REVIEW


@pytest.mark.parametrize(
    ("forward", "backward", "outcome"),
    [
        (0.8, 0.8, PairOutcome.EDGE),
        (0.2, 0.2, PairOutcome.NO_EDGE),
        (0.8, 0.2, PairOutcome.REVIEW),
        (0.79, 0.9, PairOutcome.REVIEW),
        (0.21, 0.1, PairOutcome.REVIEW),
        (0.79999995, 0.9, PairOutcome.REVIEW),
        (0.20000002, 0.1, PairOutcome.REVIEW),
    ],
)
def test_the_thresholds_are_inclusive(
    forward: float, backward: float, outcome: PairOutcome
) -> None:
    table = {(INSULTS_EN, INSULTS_ES): forward, (INSULTS_ES, INSULTS_EN): backward}
    assert by_pair(generator(table)[0].generate(CORPUS))[("a-1", "b-1")].outcome is outcome


def test_pairs_below_the_cosine_floor_are_not_candidates() -> None:
    pairs = by_pair(generator(BOTH_WAYS)[0].generate(CORPUS))
    assert not any("c-2" in pair for pair in pairs)


def test_pairs_inside_one_ruleset_are_not_candidates() -> None:
    pairs = by_pair(generator(BOTH_WAYS)[0].generate(CORPUS))
    assert ("a-1", "a-2") not in pairs


def test_each_unordered_pair_appears_once_with_the_smaller_rule_id_first() -> None:
    generated = generator(BOTH_WAYS)[0].generate(CORPUS)
    keys = [(pair.rule_a, pair.rule_b) for pair in generated]
    assert keys == sorted(set(keys))
    assert all(a < b for a, b in keys)


def test_the_scorer_gets_each_direction_with_the_full_item_text() -> None:
    rich = make_ruleset("b", [("b-1", INSULTS_ES)], language=Language.ES)
    rich = rich.model_copy(
        update={
            "rules": (make_rule("b-1", "b", INSULTS_ES, scope="Comentarios", exceptions="Citas"),)
        }
    )
    corpus = RuleCorpus(rulesets=(make_ruleset("a", [("a-1", INSULTS_EN)]), rich))
    item_es = "No insultos.\nÁmbito: Comentarios\nExcepciones: Citas"
    embedder = FakeEmbedder({INSULTS_EN: [1.0, 0.0], item_es: [1.0, 0.1]})
    candidate_generator, scorer = generator(
        {(INSULTS_EN, item_es): 0.9, (item_es, INSULTS_EN): 0.85}, embedder=embedder
    )
    (pair,) = candidate_generator.generate(corpus)
    assert scorer.calls == [((INSULTS_EN,), (item_es,)), ((item_es,), (INSULTS_EN,))]
    assert pair.outcome is PairOutcome.EDGE


def test_without_candidates_the_scorer_is_not_called() -> None:
    corpus = RuleCorpus(rulesets=(make_ruleset("a", [("a-1", INSULTS_EN)]),))
    candidate_generator, scorer = generator()
    assert candidate_generator.generate(corpus) == ()
    assert scorer.calls == []


def test_the_result_does_not_depend_on_the_order_the_rulesets_were_given_in() -> None:
    reversed_corpus = RuleCorpus(rulesets=tuple(reversed(CORPUS.rulesets)))
    first = generator(BOTH_WAYS)[0].generate(CORPUS)
    second = generator(BOTH_WAYS)[0].generate(reversed_corpus)
    assert first == second


class WrongLengthScorer:
    def entailment_probabilities(self, premises: Any, hypotheses: Any) -> NDArray[np.float32]:
        return np.zeros(len(premises) + 1, dtype=np.float32)


class OutOfRangeScorer:
    def entailment_probabilities(self, premises: Any, hypotheses: Any) -> NDArray[np.float32]:
        return np.full(len(premises), 1.5, dtype=np.float32)


@pytest.mark.parametrize(
    ("scorer", "reason"), [(WrongLengthScorer(), "expected"), (OutOfRangeScorer(), "outside")]
)
def test_a_scorer_returning_the_wrong_shape_or_range_is_reported(scorer: Any, reason: str) -> None:
    search = ExactNeighbourSearch(k=5, chunk_size=4, device=CPU, min_score=0.5)
    candidate_generator = CandidateGenerator(FakeEmbedder(VECTORS), search, scorer, SETTINGS)
    with pytest.raises(EntailmentError, match=reason):
        candidate_generator.generate(CORPUS)


def test_cross_ruleset_matches_are_found_even_when_a_ruleset_has_more_than_k_near_duplicates() -> (
    None
):
    crowded = [(f"a-{n}", f"Crowded rule {n}.") for n in range(4)]
    others = [(f"b-{n}", f"Other rule {n}.") for n in range(4)]
    corpus = RuleCorpus(rulesets=(make_ruleset("a", crowded), make_ruleset("b", others)))
    vectors = {text: [1.0, 0.01 * n, 0.0] for n, (_, text) in enumerate(crowded)}
    vectors |= {text: [0.8, 0.6, 0.01 * n] for n, (_, text) in enumerate(others)}
    candidate_generator, _ = generator(k=2, embedder=FakeEmbedder(vectors))
    pairs = candidate_generator.generate(corpus)
    assert pairs
    assert all(pair.rule_a.startswith("a-") and pair.rule_b.startswith("b-") for pair in pairs)
    assert {pair.rule_a for pair in pairs} == {"a-0", "a-1", "a-2", "a-3"}
    assert {pair.rule_b for pair in pairs} == {"b-0", "b-1", "b-2", "b-3"}
