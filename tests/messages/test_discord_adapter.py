from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from banhammer.messages import (
    DiscordDataAdapter,
    IngestReport,
    RawMessageSchema,
    SourceColumns,
    SourceFormatError,
    validate_frame,
)

SOURCE = "test/discord"
COLUMNS = SourceColumns(
    message_id="id",
    channel_id="channel_id",
    parent_message_id="reference_id",
    timestamp="timestamp",
    text="content",
    is_bot="is_bot",
)
AUTHOR_FIELDS = ("author_id", "author_name", "author_discriminator")


def source_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [101, 102, 103],
            "channel_id": [7, 7, 8],
            "reference_id": [None, 101, 555],
            "timestamp": [
                "2025-03-01T10:00:00Z",
                "2025-03-01T12:00:00+02:00",
                "2025-03-02T08:30:00Z",
            ],
            "content": ["hello there", None, "tercer mensaje"],
            "is_bot": [False, None, True],
            "author_id": [1, 2, 3],
            "author_name": ["a", "b", "c"],
            "author_discriminator": ["0001", "0002", "0003"],
        }
    )


def adapter_for(path: Path, columns: SourceColumns = COLUMNS) -> DiscordDataAdapter:
    return DiscordDataAdapter([path], columns, SOURCE)


def write_parquet(tmp_path: Path, frame: pl.DataFrame) -> Path:
    path = tmp_path / "part-0.parquet"
    frame.write_parquet(path)
    return path


def test_scan_maps_the_source_to_the_raw_schema(tmp_path: Path) -> None:
    frame = adapter_for(write_parquet(tmp_path, source_rows())).scan().collect()
    validate_frame(RawMessageSchema, frame)
    assert frame["message_id"].to_list() == ["discord:101", "discord:102", "discord:103"]
    assert frame["channel_id"].to_list() == ["discord:7", "discord:7", "discord:8"]
    assert frame["parent_message_id"].to_list() == [None, "discord:101", "discord:555"]
    assert frame["source"].unique().to_list() == [SOURCE]
    assert frame["platform"].unique().to_list() == ["discord"]
    assert frame["is_bot"].to_list() == [False, None, True]
    assert frame["text"].to_list() == ["hello there", "", "tercer mensaje"]


def test_timestamps_become_utc_microseconds(tmp_path: Path) -> None:
    frame = adapter_for(write_parquet(tmp_path, source_rows())).scan().collect()
    assert frame["timestamp"].dtype == pl.Datetime("us", "UTC")
    assert frame["timestamp"].to_list() == [
        datetime(2025, 3, 1, 10, tzinfo=UTC),
        datetime(2025, 3, 1, 10, tzinfo=UTC),
        datetime(2025, 3, 2, 8, 30, tzinfo=UTC),
    ]


def test_naive_and_aware_datetime_columns_are_read_as_utc(tmp_path: Path) -> None:
    naive = source_rows().with_columns(pl.Series("timestamp", [datetime(2025, 3, 1, 10)] * 3))
    aware = naive.with_columns(pl.col("timestamp").dt.replace_time_zone("Europe/Madrid"))
    for frame in (naive, aware):
        scanned = adapter_for(write_parquet(tmp_path, frame)).scan().collect()
        assert scanned["timestamp"].dtype == pl.Datetime("us", "UTC")
    assert scanned["timestamp"][0] == datetime(2025, 3, 1, 9, tzinfo=UTC)


def test_author_fields_never_reach_the_output(tmp_path: Path) -> None:
    frame = adapter_for(write_parquet(tmp_path, source_rows())).scan().collect()
    assert not set(AUTHOR_FIELDS) & set(frame.columns)


def test_missing_optional_columns_become_null(tmp_path: Path) -> None:
    columns = COLUMNS.model_copy(update={"parent_message_id": None, "is_bot": None})
    frame = adapter_for(write_parquet(tmp_path, source_rows()), columns).scan().collect()
    validate_frame(RawMessageSchema, frame)
    assert frame["parent_message_id"].null_count() == 3
    assert frame["is_bot"].null_count() == 3


def test_empty_parent_strings_are_null(tmp_path: Path) -> None:
    rows = source_rows().with_columns(pl.Series("reference_id", ["", "101", ""]))
    frame = adapter_for(write_parquet(tmp_path, rows)).scan().collect()
    assert frame["parent_message_id"].to_list() == [None, "discord:101", None]


def test_several_files_are_concatenated(tmp_path: Path) -> None:
    rows = source_rows()
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    rows[:2].write_parquet(first)
    rows[2:].write_parquet(second)
    frame = DiscordDataAdapter([first, second], COLUMNS, SOURCE).scan().collect()
    assert frame.height == 3


def test_a_missing_column_is_named_in_the_error(tmp_path: Path) -> None:
    path = write_parquet(tmp_path, source_rows().drop("content", "id"))
    with pytest.raises(SourceFormatError, match=r"content.*id|id.*content"):
        adapter_for(path).scan()


def test_no_files_and_unsupported_formats_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceFormatError, match="no source files"):
        DiscordDataAdapter([], COLUMNS, SOURCE).scan()
    for name, write in (
        ("b.jsonl", source_rows().write_ndjson),
        ("c.csv", source_rows().write_csv),
    ):
        path = tmp_path / name
        write(path)
        with pytest.raises(SourceFormatError, match="unsupported"):
            adapter_for(path).scan()
    parquet = write_parquet(tmp_path, source_rows())
    with pytest.raises(SourceFormatError, match="unsupported"):
        DiscordDataAdapter([parquet, tmp_path / "b.jsonl"], COLUMNS, SOURCE).scan()


def test_an_unusable_timestamp_type_is_rejected(tmp_path: Path) -> None:
    path = write_parquet(tmp_path, source_rows().with_columns(pl.Series("timestamp", [1, 2, 3])))
    with pytest.raises(SourceFormatError, match="timestamp"):
        adapter_for(path).scan()


def test_a_missing_file_is_a_source_format_error(tmp_path: Path) -> None:
    with pytest.raises(SourceFormatError, match="part-9"):
        adapter_for(tmp_path / "part-9.parquet").scan()


MENTIONS = "<@123> hi <@!456> and <@&789> in <#1011> and @plain and <@123>"


def test_user_role_and_channel_mentions_are_pseudonymized_and_plain_handles_are_kept(
    tmp_path: Path,
) -> None:
    rows = source_rows().with_columns(pl.Series("content", [MENTIONS, "no mention", None]))
    frame = adapter_for(write_parquet(tmp_path, rows)).scan().collect()
    assert frame["text"][0] == "<@user> hi <@user> and <@role> in <#channel> and @plain and <@user>"
    assert frame["text"][1] == "no mention"


def test_the_report_counts_rows_oversize_rows_and_replaced_mentions(tmp_path: Path) -> None:
    rows = source_rows().with_columns(pl.Series("content", [MENTIONS, "x" * 4000, "y" * 4001]))
    adapter = adapter_for(write_parquet(tmp_path, rows))
    assert adapter.report() == IngestReport(
        rows=2,
        dropped_oversize=1,
        mentions_replaced={"<discord-link>": 0, "<@user>": 3, "<@role>": 1, "<#channel>": 1},
    )
    texts = adapter.scan().collect()["text"]
    assert texts[1] == "x" * 4000
    assert texts.len() == 2


def test_the_scan_is_lazy_and_does_not_fail_on_unparseable_values(tmp_path: Path) -> None:
    rows = source_rows().with_columns(pl.Series("timestamp", ["2025-03-01T10:00:00Z", "junk", ""]))
    frame = adapter_for(write_parquet(tmp_path, rows)).scan().collect()
    assert frame["timestamp"].null_count() == 2


def test_unparseable_timestamps_fail_the_report_with_the_column_and_a_count_only(
    tmp_path: Path,
) -> None:
    source_value = "SECRET-not-a-timestamp"
    rows = source_rows().with_columns(
        pl.Series("timestamp", ["2025-03-01T10:00:00Z", source_value, "2025-03-01T10:00:01Z"])
    )
    with pytest.raises(SourceFormatError, match=r"'timestamp': 1 value") as raised:
        adapter_for(write_parquet(tmp_path, rows)).report()
    assert source_value not in str(raised.value)


def test_unparseable_bot_flags_fail_the_report_with_the_column_and_a_count_only(
    tmp_path: Path,
) -> None:
    source_value = "SECRET-maybe"
    rows = source_rows().with_columns(pl.Series("is_bot", ["true", source_value, "0"]))
    adapter = adapter_for(write_parquet(tmp_path, rows))
    with pytest.raises(SourceFormatError, match=r"'is_bot': 1 value") as raised:
        adapter.report()
    assert source_value not in str(raised.value)


def test_string_bot_flags_are_read_as_booleans(tmp_path: Path) -> None:
    rows = source_rows().with_columns(pl.Series("is_bot", ["True", "false", "0"]))
    adapter = adapter_for(write_parquet(tmp_path, rows))
    assert adapter.report().rows == 3
    assert adapter.scan().collect()["is_bot"].to_list() == [True, False, False]


def test_a_source_null_timestamp_is_not_blamed_on_parsing(tmp_path: Path) -> None:
    rows = source_rows().with_columns(
        pl.Series("timestamp", ["2025-03-01T10:00:00Z", None, "2025-03-01T10:00:01Z"])
    )
    assert adapter_for(write_parquet(tmp_path, rows)).report().rows == 3


def test_a_polars_error_is_re_raised_with_its_class_name_only(tmp_path: Path) -> None:
    marker = "source-value-that-must-not-leak"
    path = tmp_path / "broken.parquet"
    path.write_text(marker, encoding="utf-8")
    for call in (adapter_for(path).scan, adapter_for(path).report):
        with pytest.raises(SourceFormatError) as raised:
            call()
        assert marker not in str(raised.value)
        assert str(path) not in str(raised.value)
        assert str(raised.value).endswith("Error")
        assert raised.value.__suppress_context__


def test_discord_links_carrying_ids_are_replaced_and_counted(tmp_path: Path) -> None:
    links = (
        "https://discord.com/users/123456 and https://discord.com/channels/1/2/3 "
        "and ptb.discord.com/channels/10/20 and https://canary.discordapp.com/channels/5/6 "
        "and HTTPS://DISCORD.COM/users/9 but not https://discord.com/invite/abc "
        "and https://discord.com/channels/@me/5"
    )
    rows = source_rows().with_columns(pl.Series("content", [links, "plain", "x"]))
    adapter = adapter_for(write_parquet(tmp_path, rows))
    text = adapter.scan().collect()["text"][0]
    assert text.count("<discord-link>") == 5
    assert "123456" not in text
    assert "discord.com/invite/abc" in text
    assert adapter.report().mentions_replaced["<discord-link>"] == 5
