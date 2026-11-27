import datetime

import pytest
from pydantic import ValidationError

from banhammer.core.constants import Language
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.records import Ruleset
from rule_corpus_builders import make_rule, make_ruleset


def small_corpus() -> RuleCorpus:
    return RuleCorpus(
        rulesets=(
            make_ruleset("zeta", [("zeta-1", "No spam.")], harvest_date=datetime.date(2026, 11, 1)),
            make_ruleset(
                "alpha",
                [("alpha-2", "Be kind."), ("alpha-1", "No insults.")],
                harvest_date=datetime.date(2026, 11, 3),
            ),
            make_ruleset(
                "mid",
                [("mid-1", "Nada de spam.")],
                language=Language.ES,
                harvest_date=datetime.date(2026, 11, 1),
            ),
        )
    )


def test_rulesets_are_ordered_by_id_and_rules_keep_their_file_order() -> None:
    corpus = small_corpus()
    assert [ruleset.ruleset_id for ruleset in corpus.rulesets] == ["alpha", "mid", "zeta"]
    assert [rule.rule_id for rule in corpus.rules] == ["alpha-2", "alpha-1", "mid-1", "zeta-1"]


def test_the_corpus_does_not_depend_on_the_order_rulesets_are_given_in() -> None:
    corpus = small_corpus()
    assert RuleCorpus(rulesets=tuple(reversed(corpus.rulesets))) == corpus


def test_harvest_order_is_by_date_then_ruleset_id() -> None:
    assert small_corpus().harvest_order == ("mid", "zeta", "alpha")


def test_ruleset_of_a_rule() -> None:
    corpus = small_corpus()
    assert corpus.ruleset_of("alpha-1").ruleset_id == "alpha"
    with pytest.raises(KeyError, match="nope"):
        corpus.ruleset_of("nope")


def test_item_text_is_the_rule_alone_when_it_has_no_scope_or_exceptions() -> None:
    corpus = small_corpus()
    assert corpus.item_texts == ("Be kind.", "No insults.", "Nada de spam.", "No spam.")


def test_item_text_adds_scope_and_exceptions_with_labels_in_the_ruleset_language() -> None:
    english = make_ruleset("en-set", [("en-1", "No insults.")])
    spanish = make_ruleset("es-set", [("es-1", "No insultos.")], language=Language.ES)
    rich_en = english.model_copy(
        update={
            "rules": (
                make_rule("en-1", "en-set", "No insults.", scope="Comments", exceptions="Quotes"),
            )
        }
    )
    rich_es = spanish.model_copy(
        update={
            "rules": (
                make_rule(
                    "es-1", "es-set", "No insultos.", scope="Comentarios", exceptions="Citas"
                ),
            )
        }
    )
    corpus = RuleCorpus(rulesets=(rich_en, rich_es))
    assert corpus.item_texts == (
        "No insults.\nScope: Comments\nExceptions: Quotes",
        "No insultos.\nÁmbito: Comentarios\nExcepciones: Citas",
    )


def test_item_text_with_only_exceptions_omits_the_scope_line() -> None:
    ruleset = make_ruleset("only", [("only-1", "No insults.")])
    only = ruleset.model_copy(
        update={"rules": (make_rule("only-1", "only", "No insults.", exceptions="Quotes"),)}
    )
    assert RuleCorpus(rulesets=(only,)).item_texts == ("No insults.\nExceptions: Quotes",)


def test_an_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValidationError, match="rulesets"):
        RuleCorpus(rulesets=())


def test_a_ruleset_id_used_twice_is_rejected() -> None:
    first = make_ruleset("dup", [("dup-1", "A.")])
    second = make_ruleset("dup", [("dup-2", "B.")])
    with pytest.raises(ValidationError, match="dup"):
        RuleCorpus(rulesets=(first, second))


def test_a_rule_id_used_in_two_rulesets_is_rejected() -> None:
    first = make_ruleset("one", [("shared-1", "A.")])
    second = make_ruleset("two", [("shared-1", "B.")])
    with pytest.raises(ValidationError, match=r"shared-1.*one.*two"):
        RuleCorpus(rulesets=(first, second))


def test_rule_ids_are_listed_in_corpus_order() -> None:
    assert small_corpus().rule_ids == ("alpha-2", "alpha-1", "mid-1", "zeta-1")


def test_a_ruleset_type_is_the_record_type() -> None:
    assert all(isinstance(ruleset, Ruleset) for ruleset in small_corpus().rulesets)
