from pathlib import Path

import pytest
from pydantic import ValidationError

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.errors import SettingsError
from banhammer.core.settings import ParamsFile
from banhammer.messages import MESSAGES_SECTION, MessagesSettings
from messages_params import PINNED, UNPINNED, messages_settings, pinned_section, repository_document

REPO_ROOT = Path(__file__).resolve().parents[2]


def with_override(path: tuple[str, ...], value: object) -> dict[str, object]:
    section = pinned_section()
    target = section
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value
    return section


def test_repository_params_satisfy_the_messages_settings_once_the_revision_is_pinned() -> None:
    settings = messages_settings()
    assert settings.sampling.max_messages == 50000
    assert {"en", "es"} <= set(settings.language.detector_languages)
    assert settings.dedup.candidate_floor <= settings.dedup.translation_threshold
    assert settings.source.revision == PINNED


def test_the_repository_placeholder_revision_is_rejected_as_not_pinned() -> None:
    assert repository_document()[MESSAGES_SECTION]["source"]["revision"] in {UNPINNED, PINNED}  # type: ignore[index]
    params = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME)
    if repository_document()[MESSAGES_SECTION]["source"]["revision"] == UNPINNED:  # type: ignore[index]
        with pytest.raises(SettingsError, match="revision not pinned"):
            params.section(MESSAGES_SECTION, MessagesSettings)
    else:
        params.section(MESSAGES_SECTION, MessagesSettings)


def test_every_nested_model_rejects_a_missing_key() -> None:
    for path in [("source", "repo_id"), ("language", "min_length"), ("dedup", "k"), ("sampling",)]:
        section = pinned_section()
        target = section
        for key in path[:-1]:
            target = target[key]  # type: ignore[assignment]
        del target[path[-1]]
        with pytest.raises(ValidationError, match=path[-1]):
            MessagesSettings.model_validate(section)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="surprise"):
        MessagesSettings.model_validate(with_override(("dedup", "surprise"), 1))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source", "revision"), "main"),
        (("source", "revision"), UNPINNED),
        (("source", "allow_patterns"), []),
        (("source", "allow_patterns"), ["*"]),
        (("source", "allow_patterns"), ["*.parquet", "*.pkl"]),
        (("source", "etag_timeout"), 0),
        (("language", "min_length"), 0),
        (("language", "confidence"), 0),
        (("language", "confidence"), 1.5),
        (("language", "mixed_confidence"), 0),
        (("language", "detector_languages"), ["en", "pt"]),
        (("language", "detector_languages"), ["en", "es", "es"]),
        (("dedup", "jaccard_threshold"), 0),
        (("dedup", "jaccard_threshold"), 1.1),
        (("dedup", "num_perm"), 0),
        (("dedup", "minhash_seed"), -1),
        (("dedup", "lsh_recall_weight"), 0),
        (("dedup", "lsh_recall_weight"), 1),
        (("dedup", "shingle_size"), 0),
        (("dedup", "k"), 0),
        (("dedup", "chunk_size"), 0),
        (("dedup", "max_group_size_warning"), 0),
        (("dedup", "candidate_floor"), 0.95),
        (("calibration", "band_half_width"), 0),
        (("calibration", "top_count"), -1),
        (("sampling", "max_messages"), 0),
        (("sampling", "window_size"), 0),
        (("sampling", "strata"), 0),
        (("sampling", "max_messages"), 199),
    ],
)
def test_out_of_range_values_are_rejected(path: tuple[str, ...], value: object) -> None:
    with pytest.raises(ValidationError):
        MessagesSettings.model_validate(with_override(path, value))


def test_the_unpinned_error_says_what_to_do() -> None:
    with pytest.raises(ValidationError, match=r"revision not pinned.*commit hash"):
        MessagesSettings.model_validate(with_override(("source", "revision"), UNPINNED))


def test_columns_the_source_lacks_may_be_null() -> None:
    section = with_override(("source", "columns", "is_bot"), None)
    settings = MessagesSettings.model_validate(section)
    assert settings.source.columns.is_bot is None


def test_settings_are_frozen() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        messages_settings().sampling.strata = 3
