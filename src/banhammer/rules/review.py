"""Manual-review queues and decisions as JSON Lines files (D15, D23).

The queues are written for a reviewer, and later for the Gradio app of `annotation`; the
decisions are what the reviewer writes back. One JSON object per line, UTF-8, so a human
can read and edit both with a text editor.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

from pydantic import Field, ValidationError, model_validator

from banhammer.core.atomic_write import atomic_destination
from banhammer.rules.constants import (
    CLUSTER_ID_PATTERN,
    COSINE_DISTANCE_MAX,
    COSINE_MAX,
    COSINE_MIN,
    ENTAILMENT_DECISIONS_FILE_NAME,
    ENTAILMENT_QUEUE_FILE_NAME,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
    SPLIT_DECISIONS_FILE_NAME,
    SPLIT_QUEUE_FILE_NAME,
)
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import ReviewFileError
from banhammer.rules.records import (
    Cluster,
    RuleId,
    RulePairRecord,
    RulesRecord,
    ScoredPair,
    describe_validation_error,
)

logger = logging.getLogger(__name__)


class ReviewMember(RulesRecord):
    """One member of a cluster under split review, with the item text the reviewer reads."""

    rule_id: RuleId
    text: str = Field(min_length=1)


class EntailmentReviewItem(RulePairRecord):
    """A candidate pair the entailment model could not settle, for a human to accept or reject.

    The pair was routed on the float32 probabilities against float32 thresholds, so the rounded
    scores shown here can sit exactly on a threshold.
    """

    text_a: str = Field(min_length=1)
    text_b: str = Field(min_length=1)
    cosine: float = Field(ge=COSINE_MIN, le=COSINE_MAX)
    p_entail_ab: float = Field(ge=PROBABILITY_MIN, le=PROBABILITY_MAX)
    p_entail_ba: float = Field(ge=PROBABILITY_MIN, le=PROBABILITY_MAX)


class EntailmentDecision(RulePairRecord):
    """The reviewer's answer for one queued pair: accept it as an edge or reject it."""

    accept: bool


class SplitReviewItem(RulesRecord):
    """A cluster whose diameter exceeds the limit, for a human to keep whole or split."""

    cluster_id: str = Field(pattern=CLUSTER_ID_PATTERN)
    diameter: float = Field(ge=0, le=COSINE_DISTANCE_MAX)
    members: tuple[ReviewMember, ...] = Field(min_length=2)


class SplitDecision(RulesRecord):
    """The reviewer's answer for one queued cluster, as a partition of its members.

    One group keeps the cluster whole; several groups split it into one cluster each.
    """

    cluster_id: str = Field(pattern=CLUSTER_ID_PATTERN)
    groups: tuple[tuple[RuleId, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _groups_are_disjoint_and_non_empty(self) -> Self:
        if any(not group for group in self.groups):
            message = "every group needs at least one rule id"
            raise ValueError(message)
        counts = Counter(rule_id for group in self.groups for rule_id in group)
        repeated = sorted(rule_id for rule_id, count in counts.items() if count > 1)
        if repeated:
            message = f"rule ids placed in more than one group: {repeated}"
            raise ValueError(message)
        return self

    @property
    def splits_cluster(self) -> bool:
        """Whether the decision divides the cluster into more than one group."""
        return len(self.groups) > 1


class ReviewDecisions(RulesRecord):
    """Every decision read back from the reviewer's files."""

    entailment: tuple[EntailmentDecision, ...]
    splits: tuple[SplitDecision, ...]


class ReviewQueueBuilder:
    """Turns pairs and clusters into queue items by attaching the item texts to read."""

    def __init__(self, corpus: RuleCorpus) -> None:
        """Look item texts up in `corpus`."""
        self._texts = dict(zip(corpus.rule_ids, corpus.item_texts, strict=True))

    def entailment_items(self, pairs: Sequence[ScoredPair]) -> tuple[EntailmentReviewItem, ...]:
        """One queue item per pair, in the order given."""
        return tuple(
            EntailmentReviewItem(
                rule_a=pair.rule_a,
                rule_b=pair.rule_b,
                text_a=self._texts[pair.rule_a],
                text_b=self._texts[pair.rule_b],
                cosine=pair.cosine,
                p_entail_ab=pair.p_entail_ab,
                p_entail_ba=pair.p_entail_ba,
            )
            for pair in pairs
        )

    def split_items(self, clusters: Sequence[Cluster]) -> tuple[SplitReviewItem, ...]:
        """One queue item per cluster, in the order given."""
        return tuple(
            SplitReviewItem(
                cluster_id=cluster.cluster_id,
                diameter=cluster.diameter,
                members=tuple(
                    ReviewMember(rule_id=rule_id, text=self._texts[rule_id])
                    for rule_id in cluster.rule_ids
                ),
            )
            for cluster in clusters
        )


class ReviewFiles:
    """The queue and decision files of the manual review, under fixed names in one directory."""

    def __init__(self, directory: Path) -> None:
        """Use `directory` for the four files; it is created on the first write."""
        self._directory = directory

    def write_entailment_queue(self, items: Sequence[EntailmentReviewItem]) -> None:
        """Replace the entailment queue with `items`."""
        self._write(ENTAILMENT_QUEUE_FILE_NAME, items)

    def write_split_queue(self, items: Sequence[SplitReviewItem]) -> None:
        """Replace the split queue with `items`."""
        self._write(SPLIT_QUEUE_FILE_NAME, items)

    def read_decisions(self) -> ReviewDecisions:
        """Read both decision files; a file that does not exist yet means no decisions.

        Raises:
            ReviewFileError: A file is unreadable, a line is not a valid decision, or two
                decisions answer the same pair or cluster.
        """
        entailment = self._read(ENTAILMENT_DECISIONS_FILE_NAME, EntailmentDecision)
        splits = self._read(SPLIT_DECISIONS_FILE_NAME, SplitDecision)
        self._require_distinct(
            ENTAILMENT_DECISIONS_FILE_NAME, [(d.rule_a, d.rule_b) for d in entailment]
        )
        self._require_distinct(SPLIT_DECISIONS_FILE_NAME, [d.cluster_id for d in splits])
        return ReviewDecisions(entailment=entailment, splits=splits)

    def _write(self, name: str, items: Sequence[RulesRecord]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        lines = "".join(f"{item.model_dump_json()}\n" for item in items)
        with atomic_destination(self._directory / name) as temporary:
            temporary.write_text(lines, encoding="utf-8")
        logger.info("wrote %s: %d items", name, len(items))

    def _read[RecordT: RulesRecord](self, name: str, record: type[RecordT]) -> tuple[RecordT, ...]:
        path = self._directory / name
        if not path.exists():
            return ()
        try:
            lines = path.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError) as error:
            message = f"{path}: cannot read decisions: {error}"
            raise ReviewFileError(message) from error
        return tuple(
            self._parse(path, number, line, record)
            for number, line in enumerate(lines, start=1)
            if line.strip()
        )

    @staticmethod
    def _parse[RecordT: RulesRecord](
        path: Path, number: int, line: str, record: type[RecordT]
    ) -> RecordT:
        try:
            return record.model_validate_json(line)
        except ValidationError as error:
            message = f"{path}:{number}: {describe_validation_error(error)}"
            raise ReviewFileError(message) from error

    @staticmethod
    def _require_distinct(name: str, keys: Iterable[object]) -> None:
        repeated = sorted(str(key) for key, count in Counter(keys).items() if count > 1)
        if repeated:
            message = f"{name}: more than one decision for {repeated}"
            raise ReviewFileError(message)
