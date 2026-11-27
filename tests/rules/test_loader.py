import datetime
from pathlib import Path

import pytest

from banhammer.core.constants import Language
from banhammer.rules.constants import RuleType
from banhammer.rules.errors import RuleCorpusError, RulesError, RulesetFormatError
from banhammer.rules.loader import RuleCorpusLoader

RULE_BLOCK = """\
  - rule_id: {rule_id}
    ruleset_id: {ruleset_id}
    text: {text}
    rule_type: conduct
    discretion_flag: false
"""

HEADER = """\
ruleset_id: {ruleset_id}
platform: reddit
language: {language}
source_url: https://www.reddit.com/r/{ruleset_id}/about/rules
harvest_date: 2026-11-02
terms_note: Public rule page, quoted for research.
rules:
"""


def ruleset_text(
    ruleset_id: str = "askspain",
    rules: tuple[tuple[str, str], ...] = (("askspain-01", "No insults."),),
    language: str = "es",
) -> str:
    blocks = "".join(RULE_BLOCK.format(rule_id=i, ruleset_id=ruleset_id, text=t) for i, t in rules)
    return HEADER.format(ruleset_id=ruleset_id, language=language) + blocks


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_ruleset_file_loads_into_a_validated_ruleset(tmp_path: Path) -> None:
    path = write(tmp_path, "askspain.yaml", ruleset_text())
    ruleset = RuleCorpusLoader().load_ruleset(path)
    assert ruleset.ruleset_id == "askspain"
    assert ruleset.language is Language.ES
    assert ruleset.harvest_date == datetime.date(2026, 11, 2)
    assert ruleset.rules[0].rule_type is RuleType.CONDUCT


def test_spanish_text_with_accents_survives_loading(tmp_path: Path) -> None:
    text = ruleset_text(rules=(("askspain-01", "No se permiten insultos ni descalificaciones."),))
    path = write(tmp_path, "askspain.yaml", text.replace("insultos", "ofensas más duras"))
    assert "más duras" in RuleCorpusLoader().load_ruleset(path).rules[0].text


def test_a_directory_loads_every_yaml_file_and_ignores_other_files(tmp_path: Path) -> None:
    write(tmp_path, "b.yaml", ruleset_text("b", (("b-1", "Rule b."),), "en"))
    write(tmp_path, "a.yaml", ruleset_text("a", (("a-1", "Rule a."),), "en"))
    write(tmp_path, "README.md", "not a ruleset")
    corpus = RuleCorpusLoader().load_directory(tmp_path)
    assert [ruleset.ruleset_id for ruleset in corpus.rulesets] == ["a", "b"]


def test_a_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(RulesetFormatError, match="missing"):
        RuleCorpusLoader().load_directory(tmp_path / "missing")


def test_a_directory_without_rulesets_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "notes.txt", "nothing")
    with pytest.raises(RulesetFormatError, match=r"no \.yaml"):
        RuleCorpusLoader().load_directory(tmp_path)


def test_the_file_name_must_equal_the_ruleset_id(tmp_path: Path) -> None:
    path = write(tmp_path, "other.yaml", ruleset_text())
    with pytest.raises(RulesetFormatError, match=r"other\.yaml.*askspain"):
        RuleCorpusLoader().load_ruleset(path)


def test_a_rule_naming_another_ruleset_is_reported_with_the_file(tmp_path: Path) -> None:
    text = ruleset_text().replace(
        "    ruleset_id: askspain\n    text", "    ruleset_id: mexico\n    text"
    )
    path = write(tmp_path, "askspain.yaml", text)
    with pytest.raises(RulesetFormatError, match=r"askspain\.yaml.*mexico"):
        RuleCorpusLoader().load_ruleset(path)


def test_an_unknown_rule_type_is_reported_with_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "askspain.yaml", ruleset_text().replace("conduct", "spam"))
    with pytest.raises(RulesetFormatError, match=r"askspain\.yaml.*rule_type"):
        RuleCorpusLoader().load_ruleset(path)


def test_empty_rule_text_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path, "askspain.yaml", ruleset_text(rules=(("askspain-01", "''"),)))
    with pytest.raises(RulesetFormatError, match="text"):
        RuleCorpusLoader().load_ruleset(path)


def test_duplicate_rule_ids_in_one_file_are_reported(tmp_path: Path) -> None:
    text = ruleset_text(rules=(("askspain-01", "A."), ("askspain-01", "B.")))
    path = write(tmp_path, "askspain.yaml", text)
    with pytest.raises(RulesetFormatError, match="askspain-01"):
        RuleCorpusLoader().load_ruleset(path)


def test_a_duplicate_key_in_the_yaml_is_reported_not_silently_overridden(tmp_path: Path) -> None:
    path = write(tmp_path, "askspain.yaml", ruleset_text() + "    text: Second text.\n")
    with pytest.raises(RulesetFormatError, match=r"duplicate key 'text'"):
        RuleCorpusLoader().load_ruleset(path)


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path, "askspain.yaml", "ruleset_id: [unclosed\n")
    with pytest.raises(RulesetFormatError, match=r"askspain\.yaml"):
        RuleCorpusLoader().load_ruleset(path)


def test_a_file_that_is_not_a_mapping_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path, "askspain.yaml", "- just\n- a list\n")
    with pytest.raises(RulesetFormatError, match="mapping"):
        RuleCorpusLoader().load_ruleset(path)


def test_an_unreadable_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "askspain.yaml"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(RulesetFormatError, match=r"askspain\.yaml"):
        RuleCorpusLoader().load_ruleset(path)


def test_python_objects_in_yaml_are_not_constructed(tmp_path: Path) -> None:
    path = write(tmp_path, "askspain.yaml", "ruleset_id: !!python/object:os.system {}\n")
    with pytest.raises(RulesetFormatError):
        RuleCorpusLoader().load_ruleset(path)


def test_a_rule_id_shared_by_two_files_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", ruleset_text("a", (("shared-1", "Rule a."),), "en"))
    write(tmp_path, "b.yaml", ruleset_text("b", (("shared-1", "Rule b."),), "en"))
    with pytest.raises(RuleCorpusError, match=r"shared-1.*a.*b"):
        RuleCorpusLoader().load_directory(tmp_path)


def test_every_loader_error_is_a_rules_error(tmp_path: Path) -> None:
    path = write(tmp_path, "askspain.yaml", "- list\n")
    with pytest.raises(RulesError):
        RuleCorpusLoader().load_ruleset(path)


def test_a_yaml_merge_key_may_override_an_inherited_value(tmp_path: Path) -> None:
    text = ruleset_text().replace(
        "  - rule_id: askspain-01\n    ruleset_id: askspain\n    text: No insults.\n"
        "    rule_type: conduct\n    discretion_flag: false\n",
        "  - &base\n    rule_id: askspain-01\n    ruleset_id: askspain\n    text: No insults.\n"
        "    rule_type: conduct\n    discretion_flag: false\n"
        "  - <<: *base\n    rule_id: askspain-02\n    text: Be kind.\n",
    )
    path = write(tmp_path, "askspain.yaml", text)
    ruleset = RuleCorpusLoader().load_ruleset(path)
    assert [rule.text for rule in ruleset.rules] == ["No insults.", "Be kind."]
