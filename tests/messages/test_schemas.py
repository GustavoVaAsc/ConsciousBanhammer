import pandera.polars as pa
import polars as pl
import pytest

from banhammer.messages import (
    MessageSchemaError,
    MessagesSchema,
    RawMessageSchema,
    validate_frame,
    validate_lazy_data,
)
from banhammer.messages.constants import Column
from message_frames import messages_frame, raw_frame

MESSAGE_TEXT = "never-in-an-error-message"


def test_valid_frames_pass_and_come_back_unchanged() -> None:
    raw = raw_frame([{"message_id": "m1"}, {"message_id": "m2", "parent_message_id": "m1"}])
    assert validate_frame(RawMessageSchema, raw).equals(raw)
    messages = messages_frame([{"message_id": "m1"}, {"message_id": "m2", "dup_group_id": "m1"}])
    assert validate_frame(MessagesSchema, messages).equals(messages)


def test_a_lazy_frame_is_validated_too() -> None:
    assert (
        validate_frame(RawMessageSchema, raw_frame([{"message_id": "m1"}]).lazy()).collect().height
        == 1
    )


def test_raw_schema_rejects_duplicate_and_empty_ids() -> None:
    with pytest.raises(MessageSchemaError, match="message_id"):
        validate_frame(RawMessageSchema, raw_frame([{"message_id": "m1"}, {"message_id": "m1"}]))
    with pytest.raises(MessageSchemaError, match="message_id"):
        validate_frame(RawMessageSchema, raw_frame([{"message_id": ""}]))


def test_raw_schema_rejects_a_parent_equal_to_the_message() -> None:
    frame = raw_frame([{"message_id": "m1", "parent_message_id": "m1"}])
    with pytest.raises(MessageSchemaError, match="parent"):
        validate_frame(RawMessageSchema, frame)


def test_raw_schema_rejects_extra_columns_and_wrong_types() -> None:
    frame = raw_frame([{"message_id": "m1"}])
    with pytest.raises(MessageSchemaError, match="author"):
        validate_frame(RawMessageSchema, frame.with_columns(author=pl.lit("x")))
    with pytest.raises(MessageSchemaError, match="timestamp"):
        validate_frame(RawMessageSchema, frame.with_columns(pl.col("timestamp").cast(pl.String)))


@pytest.mark.parametrize(
    ("override", "column"),
    [
        ({"language": "fr"}, "language"),
        ({"language_confidence": 1.5}, "language_confidence"),
        ({"stratum": -1}, "stratum"),
        ({"context_source": "thread"}, "context_source"),
        ({"text": ""}, "text"),
    ],
)
def test_messages_schema_rejects_out_of_domain_values(
    override: dict[str, object], column: str
) -> None:
    frame = messages_frame([{"message_id": "m1"} | override])
    with pytest.raises(MessageSchemaError, match=column):
        validate_frame(MessagesSchema, frame)


def test_the_group_id_is_the_smallest_message_id_of_its_group() -> None:
    wrong = messages_frame(
        [{"message_id": "m1", "dup_group_id": "m2"}, {"message_id": "m2", "dup_group_id": "m2"}]
    )
    with pytest.raises(MessageSchemaError, match="dup_group_id"):
        validate_frame(MessagesSchema, wrong)
    outside = messages_frame([{"message_id": "m1", "dup_group_id": "a0"}])
    with pytest.raises(MessageSchemaError, match="dup_group_id"):
        validate_frame(MessagesSchema, outside)


def test_context_source_says_reply_link_exactly_when_the_parent_is_in_the_table() -> None:
    linked = messages_frame(
        [
            {"message_id": "m1"},
            {"message_id": "m2", "parent_message_id": "m1", "context_source": "reply_link"},
        ]
    )
    validate_frame(MessagesSchema, linked)
    for rows in (
        [{"message_id": "m1"}, {"message_id": "m2", "parent_message_id": "m1"}],
        [{"message_id": "m2", "parent_message_id": "gone", "context_source": "reply_link"}],
        [{"message_id": "m2", "context_source": "reply_link"}],
    ):
        with pytest.raises(MessageSchemaError, match="context_source"):
            validate_frame(MessagesSchema, messages_frame(rows))


def test_the_error_names_checks_and_columns_but_never_data() -> None:
    frame = messages_frame([{"message_id": "m1", "text": MESSAGE_TEXT, "language": "fr"}])
    with pytest.raises(MessageSchemaError) as raised:
        validate_frame(MessagesSchema, frame)
    assert MESSAGE_TEXT not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_schema_columns_match_the_column_enum() -> None:
    raw = set(RawMessageSchema.to_schema().columns)
    messages = set(MessagesSchema.to_schema().columns)
    assert raw | messages == {column.value for column in Column}


def test_a_lazy_frame_is_only_checked_at_schema_level_until_its_data_is_validated() -> None:
    duplicated = raw_frame([{"message_id": "m1"}, {"message_id": "m1"}]).lazy()
    validate_frame(RawMessageSchema, duplicated)
    with pytest.raises(MessageSchemaError, match="message_id"):
        validate_lazy_data(RawMessageSchema, duplicated)
    validate_lazy_data(RawMessageSchema, raw_frame([{"message_id": "m1"}]).lazy())


class ExplodingSchema(pa.DataFrameModel):
    text: str

    @pa.dataframe_check
    @classmethod
    def explodes(cls, data: pa.PolarsData) -> pl.LazyFrame:
        message = f"cannot check {data.lazyframe.collect()['text'][0]}"
        raise ValueError(message)


def test_a_check_that_raises_is_reported_by_reason_code_without_its_message() -> None:
    frame = pl.DataFrame({"text": [MESSAGE_TEXT]})
    with pytest.raises(MessageSchemaError, match="CHECK_ERROR") as raised:
        validate_frame(ExplodingSchema, frame)
    assert MESSAGE_TEXT not in str(raised.value)
    assert "None" in str(raised.value)
