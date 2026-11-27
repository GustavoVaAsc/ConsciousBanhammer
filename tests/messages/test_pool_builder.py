from collections.abc import Mapping
from pathlib import Path

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.settings import ParamsFile
from banhammer.embeddings import EMBEDDER_SECTION, EmbedderSettings
from banhammer.messages import (
    IngestReport,
    MessagePoolBuilder,
    MessagesSchema,
    MessagesSettings,
    Provenance,
    validate_frame,
)
from dedup_corpus import (
    CPU,
    E1,
    E1_EDITED,
    E1_SHOUTED,
    E2,
    E3,
    S1,
    S2,
    S3,
    dedup_settings,
    embedder,
)
from message_frames import raw_frame
from messages_params import messages_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
SPANISH_TEXTS = {S1, S2, S3}
FRENCH = "bonjour tout le monde, comment ça va"
SHORT = "ok"


class RuleDetector:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def confidences(self, text: str) -> Mapping[str, float]:
        self.seen.append(text)
        if text in SPANISH_TEXTS:
            return {"en": 0.05, "es": 0.9, "fr": 0.05}
        if text == FRENCH:
            return {"en": 0.05, "es": 0.05, "fr": 0.9}
        return {"en": 0.9, "es": 0.05, "fr": 0.05}


INGEST = IngestReport(rows=12, dropped_oversize=0, mentions_replaced={"<@user>": 0})


def real_settings(**sampling: int) -> MessagesSettings:
    base = messages_settings()
    sampling_settings = base.sampling.model_copy(update=sampling)
    dedup = dedup_settings(translation_threshold=0.85, candidate_floor=0.7)
    return base.model_copy(update={"sampling": sampling_settings, "dedup": dedup})


def builder(
    settings_: MessagesSettings, seed: int = 7, detector: RuleDetector | None = None
) -> MessagePoolBuilder:
    embedder_settings = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME).section(
        EMBEDDER_SECTION, EmbedderSettings
    )
    provenance = Provenance(seed, settings_, embedder_settings)
    return MessagePoolBuilder.assemble(provenance, embedder(), detector or RuleDetector(), CPU)


def pool() -> pl.LazyFrame:
    return raw_frame(
        [
            {"message_id": "d:01", "text": E1},
            {"message_id": "d:02", "text": E1_SHOUTED, "parent_message_id": "d:01"},
            {"message_id": "d:03", "text": S1, "parent_message_id": "d:02"},
            {"message_id": "d:04", "text": E2, "is_bot": True},
            {"message_id": "d:05", "text": "   "},
            {"message_id": "d:06", "text": SHORT},
            {"message_id": "d:07", "text": FRENCH},
            {"message_id": "d:08", "text": S2, "parent_message_id": "d:04"},
            {"message_id": "d:09", "text": E3, "parent_message_id": "d:missing"},
            {"message_id": "d:10", "text": S3, "is_bot": False},
            {"message_id": "d:11", "text": E1_EDITED, "channel_id": "d:other"},
            {
                "message_id": "d:12",
                "text": E2,
                "channel_id": "d:other",
                "parent_message_id": "d:11",
            },
        ]
    ).lazy()


def test_the_boundary_frame_satisfies_the_messages_schema() -> None:
    result = builder(real_settings()).build(pool(), INGEST)
    assert validate_frame(MessagesSchema, result.messages).height == result.messages.height


def test_bots_empty_and_non_target_language_messages_are_dropped() -> None:
    result = builder(real_settings()).build(pool(), INGEST)
    kept = set(result.messages["message_id"])
    assert kept == {"d:01", "d:02", "d:03", "d:08", "d:09", "d:10", "d:11", "d:12"}
    counts = result.manifest.counts
    assert (counts.raw, counts.dropped_empty, counts.dropped_bot) == (12, 1, 1)
    assert counts.language_dropped_in_drawn_windows == {
        "too_short": 1,
        "other_language": 1,
        "low_confidence": 0,
    }
    assert (counts.eligible, counts.language_examined_in_drawn_windows, counts.sampled) == (
        10,
        10,
        8,
    )


def test_duplicate_groups_cover_copies_and_translations_and_name_the_smallest_id() -> None:
    result = builder(real_settings()).build(pool(), INGEST)
    groups = dict(zip(result.messages["message_id"], result.messages["dup_group_id"], strict=True))
    assert groups["d:01"] == groups["d:02"] == groups["d:03"] == groups["d:11"] == "d:01"
    assert groups["d:08"] == groups["d:12"] == "d:08"
    assert groups["d:09"] == "d:09"
    assert result.manifest.counts.duplicate_groups == 4
    assert result.manifest.counts.messages_in_duplicate_groups == 6


def test_context_comes_from_a_reply_link_only_when_the_parent_is_in_the_pool() -> None:
    result = builder(real_settings()).build(pool(), INGEST)
    context = dict(
        zip(result.messages["message_id"], result.messages["context_source"], strict=True)
    )
    assert context["d:02"] == context["d:03"] == context["d:12"] == "reply_link"
    assert context["d:08"] == "none"
    assert context["d:09"] == "none"
    assert context["d:01"] == "none"


def test_the_manifest_records_seed_source_parameters_and_counts() -> None:
    settings_ = real_settings()
    manifest = builder(settings_, seed=7).build(pool(), INGEST).manifest
    assert manifest.seed == 7
    assert manifest.source == settings_.source.repo_id
    assert manifest.source_revision == settings_.source.revision
    assert manifest.settings == settings_
    assert manifest.counts.per_language == {"en": 5, "es": 3}
    assert manifest.counts.sampled == 8
    assert sum(manifest.counts.sampled_per_stratum.values()) == 8
    assert sum(manifest.counts.eligible_per_stratum.values()) == 10
    assert manifest.ingest == INGEST


def test_the_calibration_set_is_built_from_the_translation_candidates() -> None:
    result = builder(real_settings()).build(pool(), INGEST)
    assert result.calibration.columns[0] == "en_message_id"
    assert result.calibration.height > 0


def test_the_cap_limits_the_sample_and_the_build_is_reproducible() -> None:
    capped = real_settings(max_messages=4, window_size=2)
    first = builder(capped).build(pool(), INGEST)
    second = builder(capped).build(pool(), INGEST)
    assert first.messages.height <= 4
    assert first.messages.equals(second.messages)
    assert first.manifest == second.manifest


def test_the_manifest_text_holds_no_message_text() -> None:
    result = builder(real_settings()).build(pool(), INGEST)
    dumped = result.manifest.model_dump_json()
    assert not any(text in dumped for text in (E1, S1, E3))


@given(
    cap=st.integers(1, 12),
    window=st.integers(1, 5),
    strata=st.integers(1, 4),
    seed=st.integers(0, 30),
)
@settings(deadline=None, max_examples=25)
def test_whatever_the_sample_the_output_is_a_valid_boundary_with_closed_groups(
    cap: int, window: int, strata: int, seed: int
) -> None:
    result = builder(
        real_settings(max_messages=cap, window_size=window, strata=strata), seed
    ).build(pool(), INGEST)
    frame = validate_frame(MessagesSchema, result.messages)
    members = set(frame["message_id"])
    assert frame.height <= cap
    assert set(frame["dup_group_id"]) <= members
    assert result.manifest.counts.sampled == frame.height


def test_a_pool_without_any_kept_message_gives_an_empty_valid_boundary() -> None:
    raw = raw_frame([{"message_id": "d:1", "text": SHORT}, {"message_id": "d:2", "text": FRENCH}])
    result = builder(real_settings()).build(raw.lazy(), INGEST)
    assert validate_frame(MessagesSchema, result.messages).height == 0
    assert result.manifest.counts.eligible == 2
    assert result.manifest.counts.sampled == 0
    assert result.manifest.counts.duplicate_groups == 0
    assert result.calibration.height == 0


def test_the_manifest_counts_orphaned_replies_group_sizes_and_edges() -> None:
    counts = builder(real_settings()).build(pool(), INGEST).manifest.counts
    assert counts.orphaned_replies == 2
    assert counts.max_group_size == 4
    assert counts.group_size_histogram == {"1": 2, "2": 1, "3-4": 1, "5-8": 0, "9-16": 0, "17+": 0}
    assert counts.exact_edges == 1
    assert counts.near_duplicate_edges == 1
    assert counts.near_duplicate_candidates >= counts.near_duplicate_edges
    assert counts.translation_edges == 3


def test_bot_and_blank_messages_never_reach_language_identification() -> None:
    bot_text = "a distinctive message written by a bot"
    raw = raw_frame(
        [
            {"message_id": "d:1", "text": E1},
            {"message_id": "d:2", "text": bot_text, "is_bot": True},
            {"message_id": "d:3", "text": " \n\t "},
        ]
    ).lazy()
    detector = RuleDetector()
    builder(real_settings(), detector=detector).build(raw, INGEST)
    assert detector.seen == [E1]


def test_language_identification_only_runs_inside_the_drawn_windows() -> None:
    detector = RuleDetector()
    capped = real_settings(max_messages=3, window_size=3, strata=1)
    result = builder(capped, detector=detector).build(pool(), INGEST)
    assert len(detector.seen) == result.manifest.counts.language_examined_in_drawn_windows
    assert (
        result.manifest.counts.language_examined_in_drawn_windows < result.manifest.counts.eligible
    )
    assert result.messages.height <= 3
