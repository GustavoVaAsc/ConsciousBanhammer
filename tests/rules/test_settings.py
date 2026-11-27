from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.settings import ParamsFile
from banhammer.rules.constants import RULES_SECTION
from banhammer.rules.settings import NliSettings, RulesSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION = "cd" * 20

NLI_FIELDS: dict[str, Any] = {
    "model": "org/nli",
    "revision": REVISION,
    "batch_size": 8,
    "max_length": 128,
}
RULES_FIELDS: dict[str, Any] = {
    "rulesets_dir": "data/rules/rulesets",
    "output_dir": "data/rules/processed",
    "review_dir": "data/rules/review",
    "neighbours_k": 10,
    "neighbours_min_cosine": 0.5,
    "neighbours_chunk_size": 256,
    "entail_high": 0.8,
    "entail_low": 0.2,
    "diameter_max": 0.5,
    "saturation_step": 10,
    "nli": NLI_FIELDS,
}


def rules_settings(**overrides: Any) -> RulesSettings:
    return RulesSettings.model_validate(RULES_FIELDS | overrides)


def test_the_repository_params_load_and_pin_the_nli_model_to_a_commit() -> None:
    settings = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME).section(RULES_SECTION, RulesSettings)
    assert settings.nli.revision != REVISION
    assert len(settings.nli.revision) == 40
    assert settings.entail_low < settings.entail_high
    assert settings.saturation_step == 10


def test_valid_settings_are_accepted() -> None:
    settings = rules_settings()
    assert settings.nli == NliSettings.model_validate(NLI_FIELDS)
    assert settings.review_dir == Path("data/rules/review")


@pytest.mark.parametrize("field", sorted(RULES_FIELDS))
def test_every_rules_field_is_required(field: str) -> None:
    fields = dict(RULES_FIELDS)
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        RulesSettings.model_validate(fields)


@pytest.mark.parametrize("field", sorted(NLI_FIELDS))
def test_every_nli_field_is_required(field: str) -> None:
    fields = dict(NLI_FIELDS)
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        NliSettings.model_validate(fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("neighbours_k", 0),
        ("neighbours_chunk_size", 0),
        ("saturation_step", 0),
        ("neighbours_min_cosine", 1.5),
        ("neighbours_min_cosine", -1.5),
        ("entail_high", 1.5),
        ("entail_low", -0.1),
        ("diameter_max", 0.0),
        ("diameter_max", 2.5),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        rules_settings(**{field: value})


def test_the_low_entailment_threshold_must_lie_below_the_high_one() -> None:
    with pytest.raises(ValidationError, match="entail_low"):
        rules_settings(entail_low=0.8, entail_high=0.8)


@pytest.mark.parametrize("revision", ["", "main", REVISION[:-1], REVISION.upper(), REVISION + "0"])
def test_the_nli_revision_must_be_a_full_lowercase_commit_hash(revision: str) -> None:
    with pytest.raises(ValidationError, match="revision"):
        rules_settings(nli=NLI_FIELDS | {"revision": revision})


@pytest.mark.parametrize("field", ["batch_size", "max_length"])
def test_nli_sizes_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        rules_settings(nli=NLI_FIELDS | {field: 0})
