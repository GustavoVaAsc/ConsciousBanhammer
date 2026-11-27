import sys

import pytest

import banhammer
from banhammer import rules
from banhammer.rules import cross_encoder_entailment_scorer

SENTENCE_TRANSFORMERS = "sentence_transformers"
PACKAGE = "banhammer.rules"
EMBEDDINGS = "banhammer.embeddings"


def test_importing_the_package_does_not_import_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = [
        name
        for name in sys.modules
        if name.split(".")[0] == SENTENCE_TRANSFORMERS
        or name in (PACKAGE, EMBEDDINGS)
        or name.startswith((f"{PACKAGE}.", f"{EMBEDDINGS}."))
    ]
    for name in fresh:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(banhammer, "rules", rules)
    __import__(PACKAGE)
    assert not [name for name in sys.modules if name.split(".")[0] == SENTENCE_TRANSFORMERS]
    assert PACKAGE in sys.modules


def test_the_backend_is_exported_lazily_and_is_the_real_class() -> None:
    assert (
        rules.CrossEncoderEntailmentScorer
        is cross_encoder_entailment_scorer.CrossEncoderEntailmentScorer
    )


def test_an_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        rules.no_such_name  # noqa: B018


def test_every_name_in_all_resolves() -> None:
    assert all(hasattr(rules, name) for name in rules.__all__)
