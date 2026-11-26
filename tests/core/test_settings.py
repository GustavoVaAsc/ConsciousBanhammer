from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from banhammer.core.constants import CORE_SECTION, PARAMS_FILE_NAME
from banhammer.core.errors import SettingsDefinitionError, SettingsError
from banhammer.core.settings import CoreSettings, ParamsFile, SettingsModel

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_CORE = "core:\n  seed: 7\n  data_dir: data\n  models_dir: models\n"


def write_params(tmp_path: Path, text: str) -> ParamsFile:
    path = tmp_path / PARAMS_FILE_NAME
    path.write_text(text, encoding="utf-8")
    return ParamsFile(path)


def test_core_section_loads_into_a_settings_model(tmp_path: Path) -> None:
    settings = write_params(tmp_path, VALID_CORE).section(CORE_SECTION, CoreSettings)
    assert settings == CoreSettings(seed=7, data_dir=Path("data"), models_dir=Path("models"))


def test_repository_params_file_satisfies_core_settings() -> None:
    settings = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME).section(CORE_SECTION, CoreSettings)
    assert isinstance(settings.seed, int)


def test_sections_owned_by_other_subpackages_are_ignored(tmp_path: Path) -> None:
    params = write_params(tmp_path, VALID_CORE + "rules:\n  anything: 1\n")
    assert params.section(CORE_SECTION, CoreSettings).seed == 7


def test_settings_are_frozen() -> None:
    settings = CoreSettings(seed=1, data_dir=Path("a"), models_dir=Path("b"))
    with pytest.raises(ValidationError, match="frozen"):
        settings.seed = 2


def test_a_settings_model_with_a_default_is_rejected_at_class_creation() -> None:
    with pytest.raises(SettingsDefinitionError, match=r"Tuned.*retries"):

        class Tuned(SettingsModel):
            retries: int = 3


def test_a_settings_model_with_a_default_factory_is_rejected_at_class_creation() -> None:
    with pytest.raises(SettingsDefinitionError, match="names"):

        class Listed(SettingsModel):
            names: list[str] = Field(default_factory=list)


def test_a_subclass_of_a_settings_model_cannot_add_a_default() -> None:
    with pytest.raises(SettingsDefinitionError, match="seed"):

        class Derived(CoreSettings):
            seed: int = 1


def test_a_settings_model_without_defaults_is_accepted() -> None:
    class Plain(SettingsModel):
        retries: int

    assert Plain(retries=1).retries == 1


@pytest.mark.parametrize("seed", [0, 1, 2**63 - 1])
def test_seed_inside_the_bound_is_accepted(tmp_path: Path, seed: int) -> None:
    text = VALID_CORE.replace("seed: 7", f"seed: {seed}")
    assert write_params(tmp_path, text).section(CORE_SECTION, CoreSettings).seed == seed


@pytest.mark.parametrize("seed", [-1, 2**63, 2**64])
def test_seed_outside_the_bound_raises_settings_error(tmp_path: Path, seed: int) -> None:
    text = VALID_CORE.replace("seed: 7", f"seed: {seed}")
    with pytest.raises(SettingsError, match="seed"):
        write_params(tmp_path, text).section(CORE_SECTION, CoreSettings)


def test_missing_field_raises_settings_error_naming_the_file(tmp_path: Path) -> None:
    params = write_params(tmp_path, "core:\n  seed: 7\n  data_dir: data\n")
    with pytest.raises(SettingsError, match=r"(?s)params\.yaml.*models_dir"):
        params.section(CORE_SECTION, CoreSettings)


def test_unknown_key_raises_settings_error(tmp_path: Path) -> None:
    params = write_params(tmp_path, VALID_CORE + "  extra: 1\n")
    with pytest.raises(SettingsError, match="extra"):
        params.section(CORE_SECTION, CoreSettings)


def test_wrong_type_raises_settings_error(tmp_path: Path) -> None:
    params = write_params(tmp_path, VALID_CORE.replace("seed: 7", "seed: forty-two"))
    with pytest.raises(SettingsError, match="seed"):
        params.section(CORE_SECTION, CoreSettings)


def test_missing_section_raises_settings_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="section 'core' is missing"):
        write_params(tmp_path, "rules:\n  a: 1\n").section(CORE_SECTION, CoreSettings)


def test_missing_file_raises_settings_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="cannot read"):
        ParamsFile(tmp_path / PARAMS_FILE_NAME).section(CORE_SECTION, CoreSettings)


def test_file_that_is_not_utf8_raises_settings_error(tmp_path: Path) -> None:
    path = tmp_path / PARAMS_FILE_NAME
    path.write_bytes(b"core:\n  seed: \xff\xfe\n")
    with pytest.raises(SettingsError, match="cannot read"):
        ParamsFile(path).section(CORE_SECTION, CoreSettings)


def test_malformed_yaml_raises_settings_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="cannot read"):
        write_params(tmp_path, "core: [unclosed").section(CORE_SECTION, CoreSettings)


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string"])
def test_non_mapping_document_raises_settings_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(SettingsError, match="top level must be a mapping"):
        write_params(tmp_path, text).section(CORE_SECTION, CoreSettings)


def test_environment_variables_do_not_override_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEED", "99")
    monkeypatch.setenv("CORE__SEED", "99")
    assert write_params(tmp_path, VALID_CORE).section(CORE_SECTION, CoreSettings).seed == 7
