"""Exceptions of the rules subpackage."""

from banhammer.core.errors import BanhammerError


class RulesError(BanhammerError):
    """Base class of every exception the rules subpackage raises on purpose."""


class RulesetFormatError(RulesError):
    """A ruleset file cannot be read, parsed, or does not match the record format."""


class RuleCorpusError(RulesError):
    """The rulesets together break a corpus rule, e.g. a rule id used twice."""


class SchemaViolationError(RulesError):
    """A boundary artifact does not satisfy its pandera schema."""


class EntailmentError(RulesError):
    """The entailment model cannot score the pairs it was given."""


class ReviewFileError(RulesError):
    """A review queue or decisions file cannot be read or does not match its record format."""


class ReviewDecisionError(RulesError):
    """A review decision does not fit the queue it answers: unknown pair, stale cluster."""


class ReviewPendingError(RulesError):
    """Canonicalization cannot finish until the manual review queues are answered."""


class CanonicalizationError(RulesError):
    """The pairs, clusters or decisions given to canonicalization contradict the corpus."""
