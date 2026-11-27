"""Stratified window sampling of the message pool (proposal 4.6)."""

import logging
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

from banhammer.messages.constants import Column, DropReason
from banhammer.messages.errors import SamplingError
from banhammer.messages.language import LanguageFilter
from banhammer.messages.settings import SamplingSettings

logger = logging.getLogger(__name__)

WINDOW = "_window"
WINDOW_SIZE = "_window_size"
RANK = "_rank"
CHANNEL_SIZE = "_channel_size"
REQUIRED_COLUMNS = (Column.MESSAGE_ID, Column.CHANNEL_ID, Column.TIMESTAMP, Column.TEXT)


@dataclass(frozen=True)
class SampleResult:
    """The sampled messages and the counts the manifest records.

    `sampled` holds the kept messages with the language columns and a `stratum` column, ordered
    by channel, time and message id. `eligible_per_stratum` counts the candidate messages of
    each stratum, `sampled_per_stratum` the sampled ones. `examined_in_drawn_windows` is the
    number of messages language identification looked at, every message of every drawn window
    including those of windows left out because they overflowed a quota, and
    `dropped_in_drawn_windows` how many of them it dropped, by reason.
    """

    sampled: pl.DataFrame
    eligible_per_stratum: Mapping[int, int]
    sampled_per_stratum: Mapping[int, int]
    examined_in_drawn_windows: int
    dropped_in_drawn_windows: Mapping[DropReason, int]


@dataclass
class _Fill:
    kept: list[pl.DataFrame]
    examined: int
    dropped: Counter[DropReason]


class ChannelSampler:
    """Draws a capped sample of whole message windows, stratified by channel size.

    Design:

    1. The candidates are the non-blank messages by humans (the caller drops blank and bot
       messages first, so they are never sampled). Per channel they are ordered by time and cut
       into windows of `window_size` consecutive candidates. Windows are cut before language
       identification, so the language and length filters never shift a window boundary and a
       window is a stretch of the actual conversation, reply chains included.
    2. Channels are ranked by their number of candidates and cut into `strata` groups of
       (nearly) equal channel count, the deciles of channel size when `strata` is 10.
    3. The cap counts kept messages, the ones that pass language identification. Strata are
       served from the smallest, each with an equal share of the budget left, so what a small
       stratum cannot fill passes on to the larger ones.
    4. Within a stratum, windows are drawn in random order. Only the messages of drawn windows
       go through language identification (in batches that double in size), so the identifier
       sees about the cap, not the pool. It sees every message of a drawn batch, including
       windows beyond the one that overflows, and the counts say so. A window is taken whole,
       with whatever of it is kept, until the next one would exceed the stratum's share; then the
       stratum stops (and logs a warning). The sample therefore never exceeds the cap and falls
       short of a stratum's share by less than one window.

    Residual cost: the window table (one row per window) is held in memory, and the candidate
    frame is scanned and sorted once for it and again, restricted to the batch's channels, for
    every batch; a very large pool pays for those scans.

    The sampler looks only at channel, time, message id and text (through the language filter),
    never at duplicate groups. Duplicate groups are formed afterwards, on the sample.
    A message whose parent was dropped (bot, other language, outside the drawn windows) keeps
    its parent id but has no context in the output.
    """

    def __init__(
        self, settings: SamplingSettings, rng: np.random.Generator, language: LanguageFilter
    ) -> None:
        """Take the settings, a generator from `Seeder.numpy_generator` and the language filter."""
        self._settings = settings
        self._rng = rng
        self._language = language

    def sample(self, candidates: pl.LazyFrame) -> SampleResult:
        """Sample `candidates`, which needs `message_id`, `channel_id`, `timestamp` and `text`.

        Raises:
            SamplingError: A needed column is missing.
        """
        _require_columns(candidates.collect_schema().names())
        windows = self._windows(candidates)
        eligible = _totals(windows)
        remaining = self._settings.max_messages
        fills: list[tuple[int, _Fill]] = []
        ordered = sorted(eligible, key=lambda stratum: (eligible[stratum], stratum))
        for served, stratum in enumerate(ordered):
            quota = remaining // (len(ordered) - served)
            stratum_windows = windows.filter(pl.col(Column.STRATUM) == stratum)
            fill = self._fill(candidates, stratum_windows, (stratum, quota))
            remaining -= sum(frame.height for frame in fill.kept)
            fills.append((stratum, fill))
        return self._result(candidates, eligible, fills)

    def _result(
        self,
        candidates: pl.LazyFrame,
        eligible: dict[int, int],
        fills: list[tuple[int, _Fill]],
    ) -> SampleResult:
        empty = self._language.annotate(candidates.head(0).collect()).kept
        frames = [
            frame.drop(WINDOW, RANK).with_columns(pl.lit(stratum, pl.Int64).alias(Column.STRATUM))
            for stratum, fill in fills
            for frame in fill.kept
        ]
        sampled = pl.concat(
            [empty.with_columns(pl.lit(None, pl.Int64).alias(Column.STRATUM)), *frames]
        ).sort(Column.CHANNEL_ID, Column.TIMESTAMP, Column.MESSAGE_ID)
        counts = dict(sampled.group_by(Column.STRATUM).len().iter_rows())
        dropped: Counter[DropReason] = sum((fill.dropped for _, fill in fills), Counter())
        return SampleResult(
            sampled,
            eligible,
            {stratum: counts.get(stratum, 0) for stratum in eligible},
            sum(fill.examined for _, fill in fills),
            {reason: dropped[reason] for reason in DropReason},
        )

    def _windows(self, candidates: pl.LazyFrame) -> pl.DataFrame:
        numbered = self._numbered(
            candidates.select(Column.MESSAGE_ID, Column.CHANNEL_ID, Column.TIMESTAMP)
        )
        windows = (
            numbered.group_by(Column.CHANNEL_ID, WINDOW, maintain_order=True)
            .agg(pl.len().alias(WINDOW_SIZE))
            .collect()
        )
        sizes = (
            windows.group_by(Column.CHANNEL_ID)
            .agg(pl.col(WINDOW_SIZE).sum().alias(CHANNEL_SIZE))
            .sort(CHANNEL_SIZE, Column.CHANNEL_ID)
        )
        strata = sizes.select(
            Column.CHANNEL_ID,
            (pl.int_range(pl.len()) * self._settings.strata // pl.len()).alias(Column.STRATUM),
        )
        return windows.join(strata, on=Column.CHANNEL_ID, maintain_order="left")

    def _numbered(self, messages: pl.LazyFrame) -> pl.LazyFrame:
        return messages.sort(Column.CHANNEL_ID, Column.TIMESTAMP, Column.MESSAGE_ID).with_columns(
            (pl.int_range(pl.len()).over(Column.CHANNEL_ID) // self._settings.window_size).alias(
                WINDOW
            )
        )

    def _fill(
        self, candidates: pl.LazyFrame, windows: pl.DataFrame, target: tuple[int, int]
    ) -> _Fill:
        stratum, quota = target
        shuffled = windows[self._rng.permutation(windows.height)].with_row_index(RANK)
        fill = _Fill([], 0, Counter())
        remaining = quota
        for batch in _batches(shuffled, math.ceil(quota / self._settings.window_size)):
            if remaining <= 0:
                break
            fetched = self._fetch(candidates, batch)
            result = self._language.annotate(fetched)
            fill.examined += fetched.height
            fill.dropped.update(result.dropped)
            taken, used, overflow = _take_prefix(batch[RANK].to_list(), result.kept, remaining)
            fill.kept.append(result.kept.filter(pl.col(RANK).is_in(taken)))
            remaining -= used
            if overflow:
                logger.warning(
                    "stratum %d stopped on a window that would exceed its quota: %d of %d messages",
                    stratum,
                    quota - remaining,
                    quota,
                )
                break
        return fill

    def _fetch(self, candidates: pl.LazyFrame, batch: pl.DataFrame) -> pl.DataFrame:
        channels = batch.select(Column.CHANNEL_ID).unique().lazy()
        of_batch = candidates.join(channels, on=Column.CHANNEL_ID, how="semi")
        keys = batch.select(Column.CHANNEL_ID, WINDOW, RANK).lazy()
        return self._numbered(of_batch).join(keys, on=[Column.CHANNEL_ID, WINDOW]).collect()


def _batches(shuffled: pl.DataFrame, first: int) -> Iterator[pl.DataFrame]:
    start, size = 0, max(first, 1)
    while start < shuffled.height:
        yield shuffled.slice(start, size)
        start, size = start + size, size * 2


def _take_prefix(
    ranks: list[int], kept: pl.DataFrame, remaining: int
) -> tuple[list[int], int, bool]:
    kept_per_rank = dict(kept.group_by(RANK).len().iter_rows())
    taken: list[int] = []
    used = 0
    for rank in ranks:
        count = kept_per_rank.get(rank, 0)
        if used + count > remaining:
            return taken, used, True
        taken.append(rank)
        used += count
    return taken, used, False


def _require_columns(present: list[str]) -> None:
    missing = [column.value for column in REQUIRED_COLUMNS if column not in present]
    if missing:
        message = f"the frame to sample lacks the columns: {', '.join(missing)}"
        raise SamplingError(message)


def _totals(windows: pl.DataFrame) -> dict[int, int]:
    totals = windows.group_by(Column.STRATUM).agg(pl.col(WINDOW_SIZE).sum())
    return dict(totals.iter_rows())
