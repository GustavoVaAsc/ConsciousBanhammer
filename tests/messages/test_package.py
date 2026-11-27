import sys

import pytest

import banhammer
from banhammer import messages
from banhammer.messages import constants

SENTENCE_TRANSFORMERS = "sentence_transformers"
PACKAGE = "banhammer.messages"


def test_importing_the_package_does_not_import_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = [
        name
        for name in sys.modules
        if name.split(".")[0] == SENTENCE_TRANSFORMERS
        or name == PACKAGE
        or name.startswith(f"{PACKAGE}.")
        or name == "banhammer.embeddings"
        or name.startswith("banhammer.embeddings.")
    ]
    for name in fresh:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(banhammer, "messages", messages)
    __import__(PACKAGE)
    assert not [name for name in sys.modules if name.split(".")[0] == SENTENCE_TRANSFORMERS]


def test_every_name_in_all_resolves() -> None:
    assert all(hasattr(messages, name) for name in messages.__all__)
    assert len(set(messages.__all__)) == len(messages.__all__)


def test_the_language_enum_is_the_shared_core_one() -> None:
    assert not hasattr(constants, "Language")
