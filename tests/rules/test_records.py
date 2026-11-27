import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from banhammer.core.constants import Language
from banhammer.rules.constants import RuleType
from banhammer.rules.records import RuleRecord, Ruleset


def rule_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rule_id": "askspain-01",
        "ruleset_id": "askspain",
        "text": "No insults.",
        "rule_type": "conduct",
        "discretion_flag": False,
    }
    return fields | overrides


def ruleset_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "ruleset_id": "askspain",
        "platform": "reddit",
        "language": "es",
        "source_url": "https://www.reddit.com/r/askspain/about/rules",
        "harvest_date": datetime.date(2026, 11, 2),
        "terms_note": "Public rule page, quoted for research.",
        "rules": [rule_fields()],
    }
    return fields | overrides


def test_a_rule_with_only_the_required_fields_has_empty_optional_fields() -> None:
    rule = RuleRecord.model_validate(rule_fields())
    assert rule.rule_type is RuleType.CONDUCT
    assert rule.discretion_flag is False
    assert (rule.scope, rule.exceptions, rule.sanction_hint) == (None, None, None)


def test_all_eight_proposal_fields_are_kept() -> None:
    rule = RuleRecord.model_validate(
        rule_fields(
            scope="Comments", exceptions="Quoting an insult to report it", sanction_hint="ban"
        )
    )
    assert set(type(rule).model_fields) == {
        "rule_id",
        "ruleset_id",
        "text",
        "scope",
        "exceptions",
        "sanction_hint",
        "rule_type",
        "discretion_flag",
    }
    assert rule.sanction_hint == "ban"


def test_text_is_stripped() -> None:
    assert RuleRecord.model_validate(rule_fields(text="  No insults.\n")).text == "No insults."


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", ""),
        ("text", "   \n"),
        ("scope", ""),
        ("exceptions", "  "),
        ("sanction_hint", ""),
        ("rule_type", "spam"),
        ("rule_type", "Conduct"),
        ("rule_id", ""),
        ("rule_id", "Rule 1"),
        ("rule_id", "-a"),
        ("ruleset_id", "Ask Spain"),
        ("discretion_flag", "maybe"),
    ],
)
def test_invalid_rule_fields_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        RuleRecord.model_validate(rule_fields(**{field: value}))


@pytest.mark.parametrize("field", ["rule_id", "ruleset_id", "text", "rule_type", "discretion_flag"])
def test_required_rule_fields_cannot_be_omitted(field: str) -> None:
    fields = rule_fields()
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        RuleRecord.model_validate(fields)


def test_unknown_rule_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="scop"):
        RuleRecord.model_validate(rule_fields(scop="typo"))


def test_records_are_frozen() -> None:
    rule = RuleRecord.model_validate(rule_fields())
    with pytest.raises(ValidationError, match="frozen"):
        rule.text = "changed"


def test_a_ruleset_holds_its_metadata_and_rules() -> None:
    ruleset = Ruleset.model_validate(ruleset_fields())
    assert ruleset.language is Language.ES
    assert ruleset.harvest_date == datetime.date(2026, 11, 2)
    assert [rule.rule_id for rule in ruleset.rules] == ["askspain-01"]


def test_a_rule_naming_another_ruleset_is_rejected() -> None:
    other = rule_fields(ruleset_id="mexico")
    with pytest.raises(ValidationError, match=r"askspain-01.*mexico.*askspain"):
        Ruleset.model_validate(ruleset_fields(rules=[other]))


def test_a_rule_id_used_twice_in_a_ruleset_is_rejected() -> None:
    rules = [rule_fields(), rule_fields(text="Be kind.")]
    with pytest.raises(ValidationError, match="askspain-01"):
        Ruleset.model_validate(ruleset_fields(rules=rules))


def test_a_ruleset_without_rules_is_rejected() -> None:
    with pytest.raises(ValidationError, match="rules"):
        Ruleset.model_validate(ruleset_fields(rules=[]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("language", "fr"),
        ("platform", "Reddit"),
        ("platform", ""),
        ("source_url", "reddit.com/r/askspain"),
        ("source_url", "ftp://example.org/rules"),
        ("harvest_date", "not a date"),
        ("terms_note", " "),
        ("ruleset_id", "Ask Spain"),
    ],
)
def test_invalid_ruleset_metadata_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        Ruleset.model_validate(ruleset_fields(**{field: value}))


@pytest.mark.parametrize(
    "field", ["platform", "language", "source_url", "harvest_date", "terms_note"]
)
def test_provenance_cannot_be_omitted(field: str) -> None:
    fields = ruleset_fields()
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        Ruleset.model_validate(fields)
