import datetime
from collections.abc import Sequence
from typing import Any

from banhammer.core.constants import Language
from banhammer.rules.constants import RuleType
from banhammer.rules.records import RuleRecord, Ruleset


def make_rule(rule_id: str, ruleset_id: str, text: str, **overrides: Any) -> RuleRecord:
    fields: dict[str, Any] = {
        "rule_id": rule_id,
        "ruleset_id": ruleset_id,
        "text": text,
        "rule_type": RuleType.CONDUCT,
        "discretion_flag": False,
    }
    return RuleRecord.model_validate(fields | overrides)


def make_ruleset(
    ruleset_id: str,
    rules: Sequence[tuple[str, str]],
    *,
    language: Language = Language.EN,
    harvest_date: datetime.date = datetime.date(2026, 11, 1),
    platform: str = "reddit",
) -> Ruleset:
    return Ruleset(
        ruleset_id=ruleset_id,
        platform=platform,
        language=language,
        source_url=f"https://example.org/{ruleset_id}",
        harvest_date=harvest_date,
        terms_note="Public rule page.",
        rules=tuple(make_rule(rule_id, ruleset_id, text) for rule_id, text in rules),
    )
