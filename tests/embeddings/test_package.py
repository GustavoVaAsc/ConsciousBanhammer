import sys

import pytest

import banhammer
from banhammer import embeddings
from banhammer.embeddings import sentence_transformer_embedder

SENTENCE_TRANSFORMERS = "sentence_transformers"
PACKAGE = "banhammer.embeddings"


def test_importing_the_package_does_not_import_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = [
        name
        for name in sys.modules
        if name.split(".")[0] == SENTENCE_TRANSFORMERS
        or name == PACKAGE
        or name.startswith(f"{PACKAGE}.")
    ]
    for name in fresh:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(banhammer, "embeddings", embeddings)
    __import__(PACKAGE)
    assert not [name for name in sys.modules if name.split(".")[0] == SENTENCE_TRANSFORMERS]
    assert PACKAGE in sys.modules


def test_the_backend_is_exported_lazily_and_is_the_real_class() -> None:
    assert (
        embeddings.SentenceTransformerEmbedder
        is sentence_transformer_embedder.SentenceTransformerEmbedder
    )


def test_an_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        embeddings.no_such_name  # noqa: B018


def test_every_name_in_all_resolves() -> None:
    assert all(hasattr(embeddings, name) for name in embeddings.__all__)
