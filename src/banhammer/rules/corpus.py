"""The validated set of rulesets and the orderings every later stage relies on."""

from collections import Counter, defaultdict
from functools import cached_property

from pydantic import Field, field_validator

from banhammer.rules.constants import ITEM_EXCEPTIONS_LABELS, ITEM_SCOPE_LABELS
from banhammer.rules.records import RuleRecord, Ruleset, RulesRecord


class RuleCorpus(RulesRecord):
    """All rulesets of the corpus, ordered by ruleset id, rules in file order.

    Ruleset ids and rule ids are unique across the corpus. The order is fixed by the data, not
    by the order the rulesets were loaded in, so every output derived from it is reproducible.
    """

    rulesets: tuple[Ruleset, ...] = Field(min_length=1)

    @field_validator("rulesets")
    @classmethod
    def _unique_and_ordered(cls, rulesets: tuple[Ruleset, ...]) -> tuple[Ruleset, ...]:
        repeated = sorted(
            ruleset_id
            for ruleset_id, count in Counter(r.ruleset_id for r in rulesets).items()
            if count > 1
        )
        if repeated:
            message = f"ruleset ids used more than once: {repeated}"
            raise ValueError(message)
        owners = defaultdict(list)
        for ruleset in rulesets:
            for rule in ruleset.rules:
                owners[rule.rule_id].append(ruleset.ruleset_id)
        shared = {rule_id: sorted(ids) for rule_id, ids in owners.items() if len(ids) > 1}
        if shared:
            message = f"rule ids used in more than one ruleset: {shared}"
            raise ValueError(message)
        return tuple(sorted(rulesets, key=lambda ruleset: ruleset.ruleset_id))

    @cached_property
    def rules(self) -> tuple[RuleRecord, ...]:
        """Every rule, rulesets in id order, rules in file order within a ruleset."""
        return tuple(rule for ruleset in self.rulesets for rule in ruleset.rules)

    @cached_property
    def rule_ids(self) -> tuple[str, ...]:
        """The ids of `rules`, in the same order."""
        return tuple(rule.rule_id for rule in self.rules)

    @cached_property
    def harvest_order(self) -> tuple[str, ...]:
        """Ruleset ids by harvest date, ties by id: the order of the saturation curve."""
        ordered = sorted(self.rulesets, key=lambda r: (r.harvest_date, r.ruleset_id))
        return tuple(ruleset.ruleset_id for ruleset in ordered)

    @cached_property
    def item_texts(self) -> tuple[str, ...]:
        """The text an item is embedded and entailed by: rule, scope and exceptions (D15).

        Parallel to `rules`. The scope and exceptions lines carry labels in the language of
        the ruleset and are left out when the rule has none.
        """
        return tuple(
            _item_text(rule, ruleset) for ruleset in self.rulesets for rule in ruleset.rules
        )

    @cached_property
    def _ruleset_by_rule_id(self) -> dict[str, Ruleset]:
        return {rule.rule_id: ruleset for ruleset in self.rulesets for rule in ruleset.rules}

    def ruleset_of(self, rule_id: str) -> Ruleset:
        """The ruleset holding `rule_id`.

        Raises:
            KeyError: No rule has this id.
        """
        return self._ruleset_by_rule_id[rule_id]


def _item_text(rule: RuleRecord, ruleset: Ruleset) -> str:
    lines = [rule.text]
    if rule.scope is not None:
        lines.append(f"{ITEM_SCOPE_LABELS[ruleset.language]}: {rule.scope}")
    if rule.exceptions is not None:
        lines.append(f"{ITEM_EXCEPTIONS_LABELS[ruleset.language]}: {rule.exceptions}")
    return "\n".join(lines)
