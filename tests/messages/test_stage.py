import logging
from pathlib import Path

import polars as pl
import pytest
import yaml

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.errors import SettingsError
from banhammer.messages import (
    MESSAGES_SECTION,
    IngestReport,
    MessagesSchema,
    MessageStore,
    stage,
    validate_frame,
)
from dedup_corpus import E1, E2, E3, S1, S2, S3, VECTORS
from embedder_double import FakeEmbedder
from message_frames import raw_frame
from messages_params import PINNED

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXTS = [E1, E2, E3, S1, S2, S3]
ENGLISH = {E1, E2, E3}


class CorpusDetector:
    def confidences(self, text: str) -> dict[str, float]:
        english = text in ENGLISH
        return {"en": 0.9 if english else 0.05, "es": 0.05 if english else 0.9, "fr": 0.05}


def params_file(tmp_path: Path) -> Path:
    document = yaml.safe_load((REPO_ROOT / PARAMS_FILE_NAME).read_text(encoding="utf-8"))
    document[MESSAGES_SECTION]["source"]["revision"] = PINNED
    document[MESSAGES_SECTION]["raw_dir"] = str(tmp_path / "raw")
    document[MESSAGES_SECTION]["output_dir"] = str(tmp_path / "out")
    document[MESSAGES_SECTION]["dedup"]["chunk_size"] = 4
    path = tmp_path / PARAMS_FILE_NAME
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def source_file(tmp_path: Path) -> Path:
    rows = {
        "id": list(range(1, len(TEXTS) + 1)),
        "channel_id": [10, 10, 10, 11, 11, 11],
        "reference_id": [None, 1, None, None, 4, 5],
        "timestamp": [f"2025-02-01T10:0{i}:00Z" for i in range(len(TEXTS))],
        "content": [f"<@123456> {TEXTS[0]}", *TEXTS[1:]],
        "is_bot": [False] * len(TEXTS),
        "author_id": list(range(100, 100 + len(TEXTS))),
    }
    path = tmp_path / "raw" / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_ingest_then_build_write_the_boundary_manifest_and_calibration_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    params = params_file(tmp_path)
    file = source_file(tmp_path)

    def download(source: object, target: object) -> list[Path]:
        return [file]

    def load_embedder(settings: object, device: object) -> FakeEmbedder:
        return FakeEmbedder(VECTORS)

    def build_detector(languages: object) -> CorpusDetector:
        return CorpusDetector()

    monkeypatch.setattr(stage, "download_source", download)
    monkeypatch.setattr(stage.SentenceTransformerEmbedder, "from_hub", load_embedder)
    monkeypatch.setattr(stage, "LinguaLanguageDetector", build_detector)

    assert stage.main(["ingest", "--params", str(params)]) == 0
    store = MessageStore(tmp_path / "out")
    raw = store.read_raw().collect().sort("message_id")
    assert raw.height == len(TEXTS)
    assert raw["text"][0] == f"<@user> {TEXTS[0]}"
    assert store.read_ingest_report().mentions_replaced["<@user>"] == 1

    assert stage.main(["build", "--params", str(params)]) == 0
    messages = validate_frame(MessagesSchema, store.read_messages().collect())
    assert messages.height == len(TEXTS)
    assert store.read_manifest().counts.raw == len(TEXTS)
    assert store.read_manifest().ingest.rows == len(TEXTS)
    assert (tmp_path / "out" / "translation_calibration.jsonl").is_file()
    assert messages["context_source"].to_list().count("reply_link") == 3
    assert caplog.records
    assert not [text for text in TEXTS if text in caplog.text]


def test_the_command_line_accepts_only_the_two_stages() -> None:
    with pytest.raises(SystemExit):
        stage.main(["publish"])


class FrameAdapter:
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def scan(self) -> pl.LazyFrame:
        return self._frame.lazy()

    def report(self) -> IngestReport:
        return IngestReport(rows=self._frame.height, dropped_oversize=0, mentions_replaced={})


def test_ingest_accepts_any_source_adapter(tmp_path: Path) -> None:
    frame = raw_frame([{"message_id": "x:1"}, {"message_id": "x:2"}])
    store = MessageStore(tmp_path)
    assert stage.run_ingest(FrameAdapter(frame), store) == 2
    assert store.read_raw().collect().equals(frame)
    assert store.read_ingest_report().rows == 2


def test_the_repository_params_stop_ingest_until_the_dataset_revision_is_pinned(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load((REPO_ROOT / PARAMS_FILE_NAME).read_text(encoding="utf-8"))
    if document[MESSAGES_SECTION]["source"]["revision"] != "0" * 40:
        return
    with pytest.raises(SettingsError, match="revision not pinned"):
        stage.main(["ingest", "--params", str(REPO_ROOT / PARAMS_FILE_NAME)])
