from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import polars as pl

from banhammer.messages import MessagesSchema, RawMessageSchema

EPOCH = datetime(2025, 1, 1, tzinfo=UTC)
TIMESTAMP_DTYPE = pl.Datetime("us", "UTC")

RAW_DEFAULTS: dict[str, object] = {
    "source": "test/source",
    "platform": "discord",
    "channel_id": "discord:c1",
    "parent_message_id": None,
    "text": "hello there everyone",
    "is_bot": None,
}
MESSAGES_DEFAULTS: dict[str, object] = {
    "source": "test/source",
    "platform": "discord",
    "channel_id": "discord:c1",
    "parent_message_id": None,
    "text": "hello there everyone",
    "language": "en",
    "language_confidence": 0.9,
    "is_mixed": False,
    "stratum": 0,
    "context_source": "none",
}


RAW_SCHEMA = {
    "message_id": pl.String,
    "source": pl.String,
    "platform": pl.String,
    "channel_id": pl.String,
    "parent_message_id": pl.String,
    "timestamp": TIMESTAMP_DTYPE,
    "text": pl.String,
    "is_bot": pl.Boolean,
}
MESSAGES_SCHEMA = {
    **{name: dtype for name, dtype in RAW_SCHEMA.items() if name != "is_bot"},
    "language": pl.String,
    "language_confidence": pl.Float64,
    "is_mixed": pl.Boolean,
    "stratum": pl.Int64,
    "context_source": pl.String,
    "dup_group_id": pl.String,
}


def _rows(
    defaults: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    return [
        defaults | {"timestamp": EPOCH + timedelta(seconds=index)} | dict(row)
        for index, row in enumerate(rows)
    ]


def raw_frame(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    """Raw-message frame; each row needs `message_id`, the rest is defaulted."""
    return pl.DataFrame(
        _rows(RAW_DEFAULTS, rows),
        schema=RAW_SCHEMA,
    ).select(list(RawMessageSchema.to_schema().columns))


def messages_frame(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    """Messages frame; each row needs `message_id`; `dup_group_id` defaults to it."""
    filled = [{"dup_group_id": row["message_id"]} | dict(row) for row in rows]
    return pl.DataFrame(
        _rows(MESSAGES_DEFAULTS, filled),
        schema=MESSAGES_SCHEMA,
    ).select(list(MessagesSchema.to_schema().columns))
