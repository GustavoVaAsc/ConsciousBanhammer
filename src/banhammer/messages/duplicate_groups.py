"""Duplicate groups from exact, near-duplicate and translation edges (D13, D14).

A `dup_group_id` is unique per table, not global: groups are formed over the rows passed in,
and a later pass over everything that enters pairing must recompute them and validate again.
"""

import logging
from dataclasses import dataclass

import polars as pl

from banhammer.core.constants import Language
from banhammer.core.union_find import UnionFind
from banhammer.messages.constants import Column, PairColumn
from banhammer.messages.errors import DeduplicationError
from banhammer.messages.near_duplicates import NearDuplicateFinder, NearDuplicates
from banhammer.messages.settings import DedupSettings
from banhammer.messages.text_cleaning import dedup_key
from banhammer.messages.translation_candidates import TranslationCandidateFinder

logger = logging.getLogger(__name__)

KEY = "_dedup_key"
REPRESENTATIVE = "_representative"


@dataclass(frozen=True)
class EdgeCounts:
    """How many edges of each kind were merged into the groups.

    `near_candidates` is how many pairs LSH proposed; `near` is how many of them passed the
    exact Jaccard check.
    """

    exact: int
    near_candidates: int
    near: int
    translation: int


@dataclass(frozen=True)
class DuplicateGroups:
    """Duplicate groups of a message frame.

    `groups` has `message_id` and `dup_group_id` in the order of the input. `candidates` holds
    every cross-language pair at or above the candidate floor, those below t2 included, as
    `en_message_id`, `es_message_id` and `cosine`; the calibration export draws from it.
    """

    groups: pl.DataFrame
    candidates: pl.DataFrame
    edges: EdgeCounts

    def group_sizes(self) -> list[int]:
        """The number of members of each group."""
        return self.groups.group_by(Column.DUP_GROUP_ID).len()["len"].to_list()


class DuplicateGrouper:
    """Groups messages that are copies, near-copies or translations of one another.

    Edges come in three steps and all go into one union-find, so groups are transitive:

    1. exact copies, by the normalized text (`dedup_key`); one representative per text takes
       part in the next two steps, so copies cannot tie in the top-k of the neighbour search;
    2. same-language near-duplicates: Jaccard of character shingles at or above t1;
    3. English-Spanish translations: LaBSE cosine at or above t2.

    The group id is the smallest message id of the group, which does not depend on the order
    of the input rows.
    """

    def __init__(
        self,
        near_duplicates: NearDuplicateFinder,
        translations: TranslationCandidateFinder,
        settings: DedupSettings,
    ) -> None:
        """Take the two edge finders and the dedup settings (t2 is read from them)."""
        self._near = near_duplicates
        self._translations = translations
        self._settings = settings

    def group(self, frame: pl.DataFrame) -> DuplicateGroups:
        """Group the rows of `frame`, which needs `message_id`, `text` and `language` columns.

        Raises:
            DeduplicationError: `message_id` values repeat.
        """
        if not frame[Column.MESSAGE_ID].is_unique().all():
            message = "message_id values must be unique to form duplicate groups"
            raise DeduplicationError(message)
        keyed = frame.select(Column.MESSAGE_ID, Column.TEXT, Column.LANGUAGE).with_columns(
            pl.Series(KEY, [dedup_key(text) for text in frame[Column.TEXT]], dtype=pl.String)
        )
        representatives = keyed.sort(Column.MESSAGE_ID).unique(KEY, keep="first")
        exact = self._exact_edges(keyed, representatives)
        near = self._near_duplicates(representatives)
        candidates = self._translations.find(
            _language(representatives, Language.EN), _language(representatives, Language.ES)
        )
        translation = _edges(
            candidates.filter(pl.col(PairColumn.COSINE) >= self._settings.translation_threshold)
        )
        components = UnionFind(keyed[Column.MESSAGE_ID].to_list())
        components.union_all([*exact, *near.edges, *translation])
        counts = EdgeCounts(len(exact), near.candidates, len(near.edges), len(translation))
        logger.info(
            "duplicate edges: exact=%d near=%d (of %d candidates) translation=%d groups=%d",
            counts.exact,
            counts.near,
            counts.near_candidates,
            counts.translation,
            components.group_count,
        )
        group_ids = components.group_ids()
        groups = pl.DataFrame(
            {
                Column.MESSAGE_ID: frame[Column.MESSAGE_ID],
                Column.DUP_GROUP_ID: [group_ids[i] for i in frame[Column.MESSAGE_ID]],
            },
            schema={Column.MESSAGE_ID: pl.String, Column.DUP_GROUP_ID: pl.String},
        )
        result = DuplicateGroups(groups, candidates, counts)
        self._warn_of_large_groups(result)
        return result

    def _exact_edges(
        self, keyed: pl.DataFrame, representatives: pl.DataFrame
    ) -> list[tuple[str, str]]:
        heads = representatives.select(KEY, pl.col(Column.MESSAGE_ID).alias(REPRESENTATIVE))
        copies = keyed.join(heads, on=KEY).filter(
            pl.col(Column.MESSAGE_ID) != pl.col(REPRESENTATIVE)
        )
        return list(zip(copies[REPRESENTATIVE], copies[Column.MESSAGE_ID], strict=True))

    def _near_duplicates(self, representatives: pl.DataFrame) -> NearDuplicates:
        found = [
            self._near.find(frame[Column.MESSAGE_ID].to_list(), frame[KEY].to_list())
            for frame in (_language(representatives, language) for language in Language)
        ]
        return NearDuplicates(
            [edge for result in found for edge in result.edges],
            sum(result.candidates for result in found),
        )

    def _warn_of_large_groups(self, result: DuplicateGroups) -> None:
        largest = max(result.group_sizes(), default=0)
        if largest > self._settings.max_group_size_warning:
            logger.warning(
                "largest duplicate group has %d messages, above the warning size %d",
                largest,
                self._settings.max_group_size_warning,
            )


def _language(frame: pl.DataFrame, language: Language) -> pl.DataFrame:
    return frame.filter(pl.col(Column.LANGUAGE) == language)


def _edges(candidates: pl.DataFrame) -> list[tuple[str, str]]:
    return list(
        zip(
            candidates[PairColumn.EN_MESSAGE_ID],
            candidates[PairColumn.ES_MESSAGE_ID],
            strict=True,
        )
    )
