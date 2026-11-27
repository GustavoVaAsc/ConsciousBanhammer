"""Reads and writes the S2 artifacts in one output directory."""

from pathlib import Path

import polars as pl

from banhammer.messages.constants import (
    CALIBRATION_FILE_NAME,
    INGEST_REPORT_FILE_NAME,
    MANIFEST_FILE_NAME,
    MESSAGES_FILE_NAME,
    RAW_MESSAGES_FILE_NAME,
)
from banhammer.messages.manifest import IngestReport, MessagesManifest
from banhammer.messages.polars_guard import hidden_polars_errors
from banhammer.messages.schemas import (
    MessagesSchema,
    RawMessageSchema,
    validate_frame,
    validate_lazy_data,
)


class MessageStore:
    """The messages directory: raw messages and the messages boundary as Parquet.

    Frames are validated against their schema before anything is written, so a failing frame
    leaves no file behind. The manifest and the ingest report are JSON; the calibration
    candidates are JSONL, not CSV, so no cell can be read as a spreadsheet formula.
    """

    def __init__(self, directory: Path) -> None:
        """Remember the directory; it is created on the first write."""
        self._directory = directory

    def write_raw(self, frame: pl.DataFrame | pl.LazyFrame) -> Path:
        """Validate `frame` against `RawMessageSchema` and stream it to Parquet.

        A lazy frame is written with `sink_parquet`, so it is never held in memory; its values
        are validated on the written file. The partial file is removed on any exception, and a
        polars error is re-raised as `SourceFormatError` carrying only its class name.
        """
        lazy = frame.lazy()
        validate_frame(RawMessageSchema, lazy)
        path = self._path(RAW_MESSAGES_FILE_NAME)
        partial = path.with_suffix(".partial")
        try:
            with hidden_polars_errors():
                lazy.sink_parquet(partial)
                validate_lazy_data(RawMessageSchema, pl.scan_parquet(partial))
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return partial.replace(path)

    def read_raw(self) -> pl.LazyFrame:
        """Lazy scan of the raw messages."""
        return pl.scan_parquet(self._directory / RAW_MESSAGES_FILE_NAME)

    def write_messages(self, frame: pl.DataFrame) -> Path:
        """Validate `frame` against `MessagesSchema` and write the messages boundary."""
        validate_frame(MessagesSchema, frame)
        return self._write_parquet(frame, MESSAGES_FILE_NAME)

    def read_messages(self) -> pl.LazyFrame:
        """Lazy scan of the messages boundary."""
        return pl.scan_parquet(self._directory / MESSAGES_FILE_NAME)

    def write_manifest(self, manifest: MessagesManifest) -> Path:
        """Write the manifest as JSON."""
        path = self._path(MANIFEST_FILE_NAME)
        path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def read_manifest(self) -> MessagesManifest:
        """Read the manifest back."""
        text = (self._directory / MANIFEST_FILE_NAME).read_text(encoding="utf-8")
        return MessagesManifest.model_validate_json(text)

    def write_ingest_report(self, report: IngestReport) -> Path:
        """Write the ingest report as JSON."""
        path = self._path(INGEST_REPORT_FILE_NAME)
        path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def read_ingest_report(self) -> IngestReport:
        """Read the ingest report back."""
        text = (self._directory / INGEST_REPORT_FILE_NAME).read_text(encoding="utf-8")
        return IngestReport.model_validate_json(text)

    def write_calibration(self, frame: pl.DataFrame) -> Path:
        """Write the translation calibration candidates as JSON lines."""
        path = self._path(CALIBRATION_FILE_NAME)
        frame.write_ndjson(path)
        return path

    def _write_parquet(self, frame: pl.DataFrame, name: str) -> Path:
        path = self._path(name)
        frame.write_parquet(path)
        return path

    def _path(self, name: str) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        return self._directory / name
