from pathlib import Path

import polars as pl
import pytest

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.settings import ParamsFile
from banhammer.embeddings import EMBEDDER_SECTION, EmbedderSettings
from banhammer.messages import (
    IngestReport,
    MessageSchemaError,
    MessagesManifest,
    MessageStore,
    PoolCounts,
)
from message_frames import messages_frame, raw_frame
from messages_params import messages_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
INGEST = IngestReport(rows=10, dropped_oversize=1, mentions_replaced={"<@user>": 4})


def manifest() -> MessagesManifest:
    embedder = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME).section(EMBEDDER_SECTION, EmbedderSettings)
    return MessagesManifest(
        seed=42,
        source="test/source",
        source_revision="ab" * 20,
        settings=messages_settings(),
        embedder=embedder,
        ingest=INGEST,
        counts=PoolCounts(
            raw=10,
            dropped_empty=1,
            dropped_bot=2,
            eligible=7,
            eligible_per_stratum={0: 3, 1: 4},
            language_examined_in_drawn_windows=6,
            language_dropped_in_drawn_windows={"too_short": 1},
            sampled=4,
            sampled_per_stratum={0: 2, 1: 2},
            per_language={"en": 3, "es": 1},
            mixed=0,
            orphaned_replies=1,
            duplicate_groups=3,
            messages_in_duplicate_groups=2,
            max_group_size=2,
            group_size_histogram={"1": 2, "2": 1},
            exact_edges=1,
            near_duplicate_candidates=2,
            near_duplicate_edges=1,
            translation_edges=0,
        ),
    )


def test_raw_messages_round_trip_through_parquet(tmp_path: Path) -> None:
    store = MessageStore(tmp_path / "out")
    frame = raw_frame([{"message_id": "m1"}, {"message_id": "m2", "is_bot": True}])
    store.write_raw(frame)
    assert store.read_raw().collect().equals(frame)


def test_a_lazy_frame_is_streamed_to_parquet(tmp_path: Path) -> None:
    store = MessageStore(tmp_path)
    frame = raw_frame([{"message_id": "m1"}, {"message_id": "m2"}])
    path = store.write_raw(frame.lazy())
    assert path.name == "raw_messages.parquet"
    assert store.read_raw().collect().equals(frame)
    assert [file.name for file in tmp_path.iterdir()] == ["raw_messages.parquet"]


def test_messages_round_trip_through_parquet(tmp_path: Path) -> None:
    store = MessageStore(tmp_path)
    frame = messages_frame([{"message_id": "m1"}, {"message_id": "m2", "dup_group_id": "m1"}])
    store.write_messages(frame)
    assert store.read_messages().collect().equals(frame)


def test_an_invalid_frame_is_rejected_and_nothing_is_left_behind(tmp_path: Path) -> None:
    store = MessageStore(tmp_path / "out")
    with pytest.raises(MessageSchemaError):
        store.write_messages(messages_frame([{"message_id": "m1", "language": "fr"}]))
    assert not (tmp_path / "out").exists()
    duplicated = raw_frame([{"message_id": "m1"}, {"message_id": "m1"}])
    for frame in (duplicated, duplicated.lazy()):
        with pytest.raises(MessageSchemaError, match="message_id"):
            store.write_raw(frame)
        assert list((tmp_path / "out").iterdir()) == []


def test_the_manifest_round_trips_through_json(tmp_path: Path) -> None:
    store = MessageStore(tmp_path)
    store.write_manifest(manifest())
    assert store.read_manifest() == manifest()


def test_the_ingest_report_round_trips_through_json(tmp_path: Path) -> None:
    store = MessageStore(tmp_path)
    store.write_ingest_report(INGEST)
    assert store.read_ingest_report() == INGEST


def test_calibration_candidates_are_written_as_json_lines_not_csv(tmp_path: Path) -> None:
    formula = '=HYPERLINK("http://example.invalid","click"), +cmd|x'
    frame = pl.DataFrame({"en_text": [formula, "a, b"], "cosine": [0.9, 0.8]})
    path = MessageStore(tmp_path).write_calibration(frame)
    assert path.suffix == ".jsonl"
    assert pl.read_ndjson(path).equals(frame)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
