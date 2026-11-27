from pathlib import Path

import pytest
from pydantic import ValidationError

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.settings import ParamsFile
from banhammer.embeddings import EMBEDDER_SECTION, EmbedderSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION = "ab" * 20


def settings_with(**overrides: object) -> EmbedderSettings:
    fields = {"model": "org/model", "revision": REVISION, "batch_size": 4, "max_seq_length": 128}
    return EmbedderSettings.model_validate(fields | overrides)


def test_repository_params_pin_labse_to_a_commit_hash() -> None:
    settings = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME).section(EMBEDDER_SECTION, EmbedderSettings)
    assert settings.model == "sentence-transformers/LaBSE"
    assert settings.batch_size > 0
    assert settings.max_seq_length == 256


def test_settings_require_every_field() -> None:
    for field in ("model", "revision", "batch_size", "max_seq_length"):
        fields = {"model": "m", "revision": REVISION, "batch_size": 1, "max_seq_length": 1}
        del fields[field]
        with pytest.raises(ValidationError, match=field):
            EmbedderSettings.model_validate(fields)


@pytest.mark.parametrize("field", ["batch_size", "max_seq_length"])
def test_sizes_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        settings_with(**{field: 0})


@pytest.mark.parametrize(
    "revision",
    [
        "",
        "main",
        "v1.0",
        REVISION[:-1],
        REVISION + "0",
        REVISION.upper(),
        "g" * 40,
        REVISION + "\n",
    ],
)
def test_revision_must_be_a_full_lowercase_commit_hash(revision: str) -> None:
    with pytest.raises(ValidationError, match="revision"):
        settings_with(revision=revision)


def test_an_empty_model_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="model"):
        settings_with(model="")
