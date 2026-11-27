"""Adapter for `jef1056/discord-data`, read from local Parquet files."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from banhammer.messages.constants import (
    BOT_FLAG_WORDS,
    DISCORD_ID_PREFIX,
    DISCORD_PLATFORM,
    ID_SEPARATOR,
    MAX_TEXT_CHARS,
    MENTION_REPLACEMENTS,
    SOURCE_SUFFIXES,
    Column,
)
from banhammer.messages.errors import SourceFormatError
from banhammer.messages.manifest import IngestReport
from banhammer.messages.polars_guard import hidden_polars_errors
from banhammer.messages.settings import SourceColumns


class DiscordDataAdapter:
    """Maps the columns of Discord message files onto the raw-message schema.

    Only the columns named in `columns` are selected, so author fields and everything else the
    files hold never leave the adapter. The column names are configuration because they were
    written against the dataset's documentation, not against its data.

    Text is pseudonymized: user mentions `<@id>` and `<@!id>` become `<@user>`, role mentions
    `<@&id>` become `<@role>` and channel mentions `<#id>` become `<#channel>`. A plain `@word`
    is left alone because it is often not a handle and replacing it would damage the text.
    Rows whose text is longer than `MAX_TEXT_CHARS` are dropped. Everything is lazy; `report`
    makes the one pass that counts what was dropped or rewritten.
    """

    def __init__(self, files: Sequence[Path], columns: SourceColumns, source: str) -> None:
        """Remember the files (all of one format), the column mapping and the source name."""
        self._files = list(files)
        self._columns = columns
        self._source = source

    def scan(self) -> pl.LazyFrame:
        """The files' messages in the `RawMessageSchema` columns.

        Unparseable timestamps and bot flags become null instead of failing the scan, so no
        source value reaches an error message; `report` counts them and fails.

        Raises:
            SourceFormatError: There are no files, a file is missing, the format is unsupported
                or mixed, a configured column is absent, or the timestamp has an unusable type.
        """
        lazy = self._scan_files()
        with hidden_polars_errors():
            schema = lazy.collect_schema()
        self._require_columns(schema.names())
        return lazy.filter(self._within_limit()).select(
            self._namespaced(self._columns.message_id).alias(Column.MESSAGE_ID),
            pl.lit(self._source).alias(Column.SOURCE),
            pl.lit(DISCORD_PLATFORM).alias(Column.PLATFORM),
            self._namespaced(self._columns.channel_id).alias(Column.CHANNEL_ID),
            self._parent(),
            self._timestamp(schema[self._columns.timestamp]).alias(Column.TIMESTAMP),
            self._pseudonymized().alias(Column.TEXT),
            self._is_bot(schema).alias(Column.IS_BOT),
        )

    def report(self) -> IngestReport:
        """Count the rows written, the oversize rows dropped and the mentions replaced.

        Raises:
            SourceFormatError: Some timestamps or bot flags cannot be parsed; the message names
                the column and the number of failing values, never a value.
        """
        lazy = self._scan_files()
        with hidden_polars_errors():
            schema = lazy.collect_schema()
        self._require_columns(schema.names())
        kept = lazy.filter(self._within_limit())
        timestamp = pl.col(self._columns.timestamp)
        parsed = self._timestamp(schema[self._columns.timestamp])
        checks = [
            (parsed.null_count() - timestamp.null_count()).alias(self._columns.timestamp),
            *self._bot_check(schema),
        ]
        with hidden_polars_errors():
            counts = self._count(kept, checks)
        row = counts.row(0, named=True)
        for check in checks:
            self._require_parsed(check.meta.output_name(), row)
        with hidden_polars_errors():
            total = lazy.select(pl.len()).collect(engine="streaming").item()
        return IngestReport(
            rows=row["rows"],
            dropped_oversize=total - row["rows"],
            mentions_replaced={
                placeholder: row[placeholder] for _, placeholder in MENTION_REPLACEMENTS
            },
        )

    def _count(self, kept: pl.LazyFrame, checks: list[pl.Expr]) -> pl.DataFrame:
        return kept.select(
            pl.len().alias("rows"),
            *(
                self._text().str.count_matches(pattern.pattern).sum().alias(placeholder)
                for pattern, placeholder in MENTION_REPLACEMENTS
            ),
            *checks,
        ).collect(engine="streaming")

    def _require_parsed(self, column: str, row: dict[str, int]) -> None:
        if row[column]:
            message = f"column {column!r}: {row[column]} value(s) cannot be parsed"
            raise SourceFormatError(message)

    def _bot_check(self, schema: pl.Schema) -> list[pl.Expr]:
        name = self._columns.is_bot
        if name is None:
            return []
        return [(self._is_bot(schema).null_count() - pl.col(name).null_count()).alias(name)]

    def _scan_files(self) -> pl.LazyFrame:
        if not self._files:
            message = "no source files were given"
            raise SourceFormatError(message)
        missing = [path.name for path in self._files if not path.is_file()]
        if missing:
            message = f"source files not found: {', '.join(missing)}"
            raise SourceFormatError(message)
        unsupported = sorted({path.suffix for path in self._files} - set(SOURCE_SUFFIXES))
        if unsupported:
            message = f"unsupported source format {unsupported}; expected {SOURCE_SUFFIXES}"
            raise SourceFormatError(message)
        return pl.scan_parquet(self._files)

    def _required_names(self) -> list[str]:
        optional = (self._columns.parent_message_id, self._columns.is_bot)
        wanted = [
            self._columns.message_id,
            self._columns.channel_id,
            self._columns.timestamp,
            self._columns.text,
        ]
        return wanted + [name for name in optional if name is not None]

    def _require_columns(self, present: list[str]) -> None:
        absent = [name for name in self._required_names() if name not in present]
        if absent:
            message = f"source files lack the columns: {', '.join(absent)}"
            raise SourceFormatError(message)

    def _text(self) -> pl.Expr:
        return pl.col(self._columns.text).cast(pl.String).fill_null("")

    def _within_limit(self) -> pl.Expr:
        return self._text().str.len_chars() <= MAX_TEXT_CHARS

    def _pseudonymized(self) -> pl.Expr:
        text = self._text()
        for pattern, placeholder in MENTION_REPLACEMENTS:
            text = text.str.replace_all(pattern.pattern, placeholder, literal=False)
        return text

    def _namespaced(self, column: str) -> pl.Expr:
        return pl.lit(f"{DISCORD_ID_PREFIX}{ID_SEPARATOR}") + pl.col(column).cast(pl.String)

    def _parent(self) -> pl.Expr:
        name = self._columns.parent_message_id
        if name is None:
            return pl.lit(None, dtype=pl.String).alias(Column.PARENT_MESSAGE_ID)
        text = pl.col(name).cast(pl.String)
        present = text.is_not_null() & (text != "")
        return pl.when(present).then(self._namespaced(name)).alias(Column.PARENT_MESSAGE_ID)

    def _is_bot(self, schema: pl.Schema) -> pl.Expr:
        name = self._columns.is_bot
        if name is None:
            return pl.lit(None, dtype=pl.Boolean)
        column = pl.col(name)
        if schema[name] == pl.String:
            return column.str.to_lowercase().replace_strict(
                BOT_FLAG_WORDS, default=None, return_dtype=pl.Boolean
            )
        return column.cast(pl.Boolean, strict=False)

    def _timestamp(self, dtype: pl.DataType) -> pl.Expr:
        column = pl.col(self._columns.timestamp)
        if dtype == pl.String:
            return column.str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            return column.dt.replace_time_zone("UTC").dt.cast_time_unit("us")
        if isinstance(dtype, pl.Datetime):
            return column.dt.convert_time_zone("UTC").dt.cast_time_unit("us")
        message = f"timestamp column {self._columns.timestamp!r} has unsupported type {dtype}"
        raise SourceFormatError(message)
