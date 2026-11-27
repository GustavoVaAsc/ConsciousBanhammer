"""The manifest written next to the messages boundary: seed, parameters, counts, source."""

from pydantic import BaseModel, ConfigDict

from banhammer.embeddings import EmbedderSettings
from banhammer.messages.settings import MessagesSettings


class IngestReport(BaseModel):
    """What ingest did to the source besides mapping columns.

    `rows` is the number of raw messages written, `dropped_oversize` the source rows dropped for
    exceeding `MAX_TEXT_CHARS`, and `mentions_replaced` the number of mentions rewritten per
    placeholder, e.g. `{"<@user>": 12}`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int
    dropped_oversize: int
    mentions_replaced: dict[str, int]


class PoolCounts(BaseModel):
    """How many messages each step of the pool construction kept and dropped.

    `eligible` is the number of non-blank messages by humans, from which windows are cut;
    `language_examined_in_drawn_windows` is how many of them language identification looked at:
    every message of every drawn window, including windows a stratum did not take because they
    overflowed its quota. `language_dropped_in_drawn_windows`, keyed by `DropReason`, is how
    many of those it dropped. The per-stratum counts are keyed by channel size decile:
    `eligible_per_stratum` before sampling, `sampled_per_stratum` after. `orphaned_replies`
    counts sampled messages whose parent is set but not in the output. Duplicate-group counts
    refer to the sample; `group_size_histogram` maps a size bucket to its number of groups.
    `near_duplicate_candidates` are the pairs LSH proposed, of which `near_duplicate_edges`
    passed the exact Jaccard check.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: int
    dropped_empty: int
    dropped_bot: int
    eligible: int
    eligible_per_stratum: dict[int, int]
    language_examined_in_drawn_windows: int
    language_dropped_in_drawn_windows: dict[str, int]
    sampled: int
    sampled_per_stratum: dict[int, int]
    per_language: dict[str, int]
    mixed: int
    orphaned_replies: int
    duplicate_groups: int
    messages_in_duplicate_groups: int
    max_group_size: int
    group_size_histogram: dict[str, int]
    exact_edges: int
    near_duplicate_candidates: int
    near_duplicate_edges: int
    translation_edges: int


class MessagesManifest(BaseModel):
    """Provenance of one messages build (proposal 4.6: record the seed and a manifest)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    source: str
    source_revision: str
    settings: MessagesSettings
    embedder: EmbedderSettings
    ingest: IngestReport
    counts: PoolCounts
