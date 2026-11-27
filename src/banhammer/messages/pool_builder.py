"""Assembly of the messages boundary from raw messages.

`dup_group_id` is unique per table, not global: groups are formed on the sampled messages
only, and the id is the smallest message id of the group. A later pass over everything that
enters pairing (this pool, the Spanish pool, labelled corpora) must recompute duplicate groups
over the union and validate them again; it must not treat an S2 `dup_group_id` as global.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import polars as pl
import torch

from banhammer.core.seeding import Seeder
from banhammer.embeddings import Embedder, EmbedderSettings, ExactNeighbourSearch
from banhammer.messages.calibration import CalibrationSetBuilder
from banhammer.messages.constants import (
    CALIBRATION_STREAM,
    GROUP_SIZE_BUCKET_BOUNDS,
    SAMPLING_STREAM,
    Column,
    ContextSource,
)
from banhammer.messages.duplicate_groups import DuplicateGrouper, DuplicateGroups
from banhammer.messages.language import LanguageDetector, LanguageFilter
from banhammer.messages.manifest import IngestReport, MessagesManifest, PoolCounts
from banhammer.messages.near_duplicates import NearDuplicateFinder
from banhammer.messages.sampling import ChannelSampler, SampleResult
from banhammer.messages.schemas import MessagesSchema, RawMessageSchema, validate_frame
from banhammer.messages.settings import MessagesSettings
from banhammer.messages.translation_candidates import TranslationCandidateFinder

MESSAGE_COLUMNS = tuple(MessagesSchema.to_schema().columns)


@dataclass(frozen=True)
class Provenance:
    """What the manifest records about a build besides counts: the seed and the settings used."""

    seed: int
    settings: MessagesSettings
    embedder: EmbedderSettings


@dataclass(frozen=True)
class PoolBuild:
    """The messages boundary frame, its manifest and the translation calibration candidates."""

    messages: pl.DataFrame
    manifest: MessagesManifest
    calibration: pl.DataFrame


class MessagePoolBuilder:
    """Builds the messages boundary (D28) from the raw messages of a source.

    The steps, in order, and what each leaves out:

    1. blank messages, then bot messages (a missing bot flag counts as human), filtered lazily;
    2. stratified window sampling up to the cap (proposal 4.6), which cuts windows over the
       remaining messages and runs language identification (D19) only inside the drawn windows;
    3. duplicate groups (D13), formed on the sample.

    Duplicate groups are formed after sampling, over exactly the sampled messages, so sampling
    can never split a group and every `dup_group_id` names a message of the output. A group the
    full pool would have is cut down to its sampled members. `context_source` is `reply_link`
    when the parent message is itself in the output (D16), `none` otherwise, so a parent dropped
    as bot, wrong language or unsampled leaves its replies without context; the manifest counts
    those orphaned replies.
    """

    def __init__(
        self,
        sampler: ChannelSampler,
        grouper: DuplicateGrouper,
        calibration: CalibrationSetBuilder,
        provenance: Provenance,
    ) -> None:
        """Take the three steps, all injected, and what the manifest records."""
        self._sampler = sampler
        self._grouper = grouper
        self._calibration = calibration
        self._provenance = provenance

    @classmethod
    def assemble(
        cls,
        provenance: Provenance,
        embedder: Embedder,
        detector: LanguageDetector,
        device: torch.device,
    ) -> Self:
        """Wire the steps from the settings in `provenance`.

        The sampler and the calibration builder draw from named streams of `Seeder` seeded with
        `provenance.seed`; the translation search runs on `device` with the candidate floor as
        its score floor.
        """
        settings = provenance.settings
        seeder = Seeder(provenance.seed)
        search = ExactNeighbourSearch(
            k=settings.dedup.k,
            chunk_size=settings.dedup.chunk_size,
            device=device,
            min_score=settings.dedup.candidate_floor,
        )
        return cls(
            ChannelSampler(
                settings.sampling,
                seeder.numpy_generator(SAMPLING_STREAM),
                LanguageFilter(detector, settings.language),
            ),
            DuplicateGrouper(
                NearDuplicateFinder(settings.dedup),
                TranslationCandidateFinder(embedder, search),
                settings.dedup,
            ),
            CalibrationSetBuilder(
                settings.calibration,
                settings.dedup.translation_threshold,
                seeder.numpy_generator(CALIBRATION_STREAM),
            ),
            provenance,
        )

    def build(self, raw: pl.LazyFrame, ingest: IngestReport) -> PoolBuild:
        """Run the steps over `raw`, which must satisfy `RawMessageSchema`.

        `ingest` is the report ingest wrote; the manifest carries it. The schema is checked at
        column level only here, its values were checked when the raw messages were written.

        Raises:
            MessageSchemaError: `raw` does not have the columns and types of `RawMessageSchema`.
        """
        validate_frame(RawMessageSchema, raw)
        blank = pl.col(Column.TEXT).str.strip_chars() == ""
        bot = pl.col(Column.IS_BOT).fill_null(value=False)
        humans = raw.filter(~blank & ~bot)
        funnel = raw.select(
            pl.len().alias("raw"), blank.sum().alias("blank"), (~blank & bot).sum().alias("bot")
        ).collect(engine="streaming")
        sample = self._sampler.sample(humans)
        grouped = self._grouper.group(sample.sampled)
        messages = _assemble(sample.sampled, grouped)
        calibration = self._calibration.build(
            grouped.candidates, messages.select(Column.MESSAGE_ID, Column.TEXT)
        )
        trace = _Trace(funnel.row(0, named=True), sample, grouped)
        return PoolBuild(messages, self._manifest(trace, messages, ingest), calibration)

    def _manifest(
        self, trace: "_Trace", messages: pl.DataFrame, ingest: IngestReport
    ) -> MessagesManifest:
        settings = self._provenance.settings
        return MessagesManifest(
            seed=self._provenance.seed,
            source=settings.source.repo_id,
            source_revision=settings.source.revision,
            settings=settings,
            embedder=self._provenance.embedder,
            ingest=ingest,
            counts=trace.counts(messages),
        )


@dataclass(frozen=True)
class _Trace:
    funnel: dict[str, int]
    sample: SampleResult
    grouped: DuplicateGroups

    def counts(self, messages: pl.DataFrame) -> PoolCounts:
        sizes = self.grouped.group_sizes()
        edges = self.grouped.edges
        return PoolCounts(
            raw=self.funnel["raw"],
            dropped_empty=self.funnel["blank"],
            dropped_bot=self.funnel["bot"],
            eligible=sum(self.sample.eligible_per_stratum.values()),
            eligible_per_stratum=dict(self.sample.eligible_per_stratum),
            language_examined_in_drawn_windows=self.sample.examined_in_drawn_windows,
            language_dropped_in_drawn_windows={
                reason.value: n for reason, n in self.sample.dropped_in_drawn_windows.items()
            },
            sampled=messages.height,
            sampled_per_stratum=dict(self.sample.sampled_per_stratum),
            per_language=dict(messages.group_by(Column.LANGUAGE).len().iter_rows()),
            mixed=int(messages[Column.IS_MIXED].sum()),
            orphaned_replies=messages.filter(
                pl.col(Column.PARENT_MESSAGE_ID).is_not_null()
                & (pl.col(Column.CONTEXT_SOURCE) == ContextSource.NONE)
            ).height,
            duplicate_groups=len(sizes),
            messages_in_duplicate_groups=sum(size for size in sizes if size > 1),
            max_group_size=max(sizes, default=0),
            group_size_histogram=_size_histogram(sizes),
            exact_edges=edges.exact,
            near_duplicate_candidates=edges.near_candidates,
            near_duplicate_edges=edges.near,
            translation_edges=edges.translation,
        )


def _size_histogram(sizes: Iterable[int]) -> dict[str, int]:
    labels = _bucket_labels()
    counts = Counter(_bucket_label(size) for size in sizes)
    return {label: counts[label] for label in labels}


def _bucket_label(size: int) -> str:
    lower = 1
    for bound in GROUP_SIZE_BUCKET_BOUNDS:
        if size <= bound:
            return str(bound) if bound == lower else f"{lower}-{bound}"
        lower = bound + 1
    return f"{lower}+"


def _bucket_labels() -> list[str]:
    bounds = (*GROUP_SIZE_BUCKET_BOUNDS, GROUP_SIZE_BUCKET_BOUNDS[-1] + 1)
    return [_bucket_label(bound) for bound in bounds]


def _assemble(sampled: pl.DataFrame, grouped: DuplicateGroups) -> pl.DataFrame:
    parent_is_present = pl.col(Column.PARENT_MESSAGE_ID).is_in(pl.col(Column.MESSAGE_ID).implode())
    return (
        sampled.join(grouped.groups, on=Column.MESSAGE_ID, how="left", maintain_order="left")
        .with_columns(
            pl.when(parent_is_present)
            .then(pl.lit(ContextSource.REPLY_LINK))
            .otherwise(pl.lit(ContextSource.NONE))
            .alias(Column.CONTEXT_SOURCE)
        )
        .select(MESSAGE_COLUMNS)
    )
