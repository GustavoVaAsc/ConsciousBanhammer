from pathlib import Path

import pytest
import torch

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.errors import SettingsError
from banhammer.rules import stage_atomize
from banhammer.rules.constants import RULES_FILE_NAME, RULESETS_FILE_NAME
from banhammer.rules.stage_runtime import StageSettings, select_device

REPO_ROOT = Path(__file__).resolve().parents[2]

RULESET = """\
ruleset_id: one
platform: reddit
language: en
source_url: https://example.org/one
harvest_date: 2026-11-01
terms_note: Public rule page.
rules:
  - rule_id: one-1
    ruleset_id: one
    text: No insults.
    rule_type: conduct
    discretion_flag: false
"""


def test_the_stage_settings_load_the_three_sections_of_the_repository_params() -> None:
    settings = StageSettings.load(REPO_ROOT / PARAMS_FILE_NAME)
    assert settings.rules.nli.model.startswith("MoritzLaurer/")
    assert settings.embedder.model == "sentence-transformers/LaBSE"
    assert settings.core.seed >= 0


def test_a_missing_section_is_reported(tmp_path: Path) -> None:
    params = tmp_path / PARAMS_FILE_NAME
    params.write_text("core:\n  seed: 1\n  data_dir: d\n  models_dir: m\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="rules"):
        StageSettings.load(params)


@pytest.mark.parametrize(("cuda", "expected"), [(True, "cuda"), (False, "cpu")])
def test_the_device_is_cuda_only_when_available(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert select_device() == torch.device(expected)


def test_the_atomize_entry_point_reads_params_from_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    params = (REPO_ROOT / PARAMS_FILE_NAME).read_text(encoding="utf-8")
    (tmp_path / PARAMS_FILE_NAME).write_text(params, encoding="utf-8")
    configured = StageSettings.load(REPO_ROOT / PARAMS_FILE_NAME).rules
    rulesets = tmp_path / configured.rulesets_dir
    rulesets.mkdir(parents=True)
    (rulesets / "one.yaml").write_text(RULESET, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    stage_atomize.main()
    output = tmp_path / configured.output_dir
    assert (output / RULES_FILE_NAME).is_file()
    assert (output / RULESETS_FILE_NAME).is_file()
