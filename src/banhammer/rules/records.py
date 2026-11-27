"""Pydantic records of the rule corpus and of its canonicalization (D28).

The rule field names are those of proposal 4.1. A `Ruleset` is also the on-disk format of one
ruleset file, so every check that a hand-written file needs lives in these models.
"""

import datetime
from collections import Counter
from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from banhammer.core.constants import Language
from banhammer.rules.constants import (
    CLUSTER_ID_PATTERN,
    CLUSTER_ID_PREFIX,
    COSINE_DISTANCE_MAX,
    COSINE_MAX,
    COSINE_MIN,
    PLATFORM_PATTERN,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
    RULE_ID_PATTERN,
    RULESET_ID_PATTERN,
    SOURCE_URL_PATTERN,
    ClusterResolution,
    PairOutcome,
    RuleType,
)

NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RuleId = Annotated[str, StringConstraints(pattern=RULE_ID_PATTERN)]


class RulesRecord(BaseModel):
    """Base of the records of this subpackage: frozen, no unknown keys, text stripped."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class RuleRecord(RulesRecord):
    """One atomic rule item of a ruleset, with the eight fields of proposal 4.1.

    `scope`, `exceptions` and `sanction_hint` are absent when the rule states none. The
    sanction hint is free text for now; the policy engine (S6) decides how it reads it.
    """

    rule_id: str = Field(pattern=RULE_ID_PATTERN)
    ruleset_id: str = Field(pattern=RULESET_ID_PATTERN)
    text: NonBlankText
    scope: NonBlankText | None = None
    exceptions: NonBlankText | None = None
    sanction_hint: NonBlankText | None = None
    rule_type: RuleType
    discretion_flag: bool


class Ruleset(RulesRecord):
    """The rules of one community with the provenance of the harvest.

    `platform` and `language` stratify the held-out split (D11); `source_url`, `harvest_date`
    and `terms_note` record provenance and terms (H14, proposal 4.6).
    """

    ruleset_id: str = Field(pattern=RULESET_ID_PATTERN)
    platform: str = Field(pattern=PLATFORM_PATTERN)
    language: Language
    source_url: str = Field(pattern=SOURCE_URL_PATTERN)
    harvest_date: datetime.date
    terms_note: NonBlankText
    rules: tuple[RuleRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _rules_belong_to_this_ruleset_once(self) -> Self:
        strangers = {
            rule.rule_id: rule.ruleset_id
            for rule in self.rules
            if rule.ruleset_id != self.ruleset_id
        }
        if strangers:
            message = f"rules name another ruleset than {self.ruleset_id!r}: {strangers}"
            raise ValueError(message)
        repeated = sorted(
            rule_id
            for rule_id, count in Counter(r.rule_id for r in self.rules).items()
            if count > 1
        )
        if repeated:
            message = f"rule ids used more than once in {self.ruleset_id!r}: {repeated}"
            raise ValueError(message)
        return self


class RulePairRecord(RulesRecord):
    """Base of the records about an unordered pair of rules, stored with the smaller id first."""

    rule_a: RuleId
    rule_b: RuleId

    @model_validator(mode="after")
    def _smaller_rule_id_first(self) -> Self:
        if self.rule_a >= self.rule_b:
            message = f"rule_a must sort before rule_b, got {self.rule_a!r} and {self.rule_b!r}"
            raise ValueError(message)
        return self


class ScoredPair(RulePairRecord):
    """A candidate pair of rules with both entailment probabilities and what they decided."""

    cosine: float = Field(ge=COSINE_MIN, le=COSINE_MAX)
    p_entail_ab: float = Field(ge=PROBABILITY_MIN, le=PROBABILITY_MAX)
    p_entail_ba: float = Field(ge=PROBABILITY_MIN, le=PROBABILITY_MAX)
    outcome: PairOutcome


class Cluster(RulesRecord):
    """A group of equivalent rules, identified by its smallest member.

    `rule_ids` are sorted and unique, and `cluster_id` is `CLUSTER_ID_PREFIX` plus the first
    of them, so the id depends only on the membership. `diameter` is the largest pairwise
    cosine distance between members, and `resolution` says how the diameter check passed.
    """

    cluster_id: str = Field(pattern=CLUSTER_ID_PATTERN)
    rule_ids: tuple[RuleId, ...] = Field(min_length=1)
    diameter: float = Field(ge=0, le=COSINE_DISTANCE_MAX)
    resolution: ClusterResolution

    @model_validator(mode="after")
    def _id_names_the_smallest_member(self) -> Self:
        if list(self.rule_ids) != sorted(set(self.rule_ids)):
            message = f"rule_ids must be sorted and unique, got {list(self.rule_ids)}"
            raise ValueError(message)
        if self.cluster_id != cluster_id_of(self.rule_ids):
            message = f"cluster_id {self.cluster_id!r} does not name its smallest member"
            raise ValueError(message)
        return self


class SaturationPoint(RulesRecord):
    """One point of the saturation curve: clusters after `rulesets_added` rulesets (D15)."""

    rulesets_added: int = Field(ge=1)
    rule_count: int = Field(ge=1)
    cluster_count: int = Field(ge=1)
    new_clusters: int = Field(ge=0)


def cluster_id_of(rule_ids: Sequence[str]) -> str:
    """The id of the cluster with these members: the prefix and its smallest rule id."""
    return f"{CLUSTER_ID_PREFIX}{min(rule_ids)}"


def describe_validation_error(error: ValidationError) -> str:
    """One line per failure, `location: message`, without echoing the offending input."""
    return "; ".join(
        f"{'.'.join(str(part) for part in failure['loc']) or 'record'}: {failure['msg']}"
        for failure in error.errors(include_url=False, include_input=False)
    )
