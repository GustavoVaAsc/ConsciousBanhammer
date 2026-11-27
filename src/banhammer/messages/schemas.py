"""Pandera schemas of the raw-message frame and of the messages boundary (D28)."""

import pandera.errors
import pandera.polars as pa
import polars as pl
from pandera.config import ValidationDepth, config_context
from pandera.errors import SchemaErrorReason

from banhammer.core.constants import Language
from banhammer.messages.constants import Column, ContextSource
from banhammer.messages.errors import MessageSchemaError

NON_EMPTY = {"min_value": 1}
NAME_ONLY_REASONS = frozenset(
    {
        SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
        SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
        SchemaErrorReason.WRONG_DATATYPE,
    }
)
UTC_MICROSECONDS = pl.Datetime("us", "UTC")


class RawMessageSchema(pa.DataFrameModel):
    """Messages as a source adapter delivers them: no author fields, namespaced ids."""

    message_id: str = pa.Field(unique=True, str_length=NON_EMPTY)
    source: str = pa.Field(str_length=NON_EMPTY)
    platform: str = pa.Field(str_length=NON_EMPTY)
    channel_id: str = pa.Field(str_length=NON_EMPTY)
    parent_message_id: str | None = pa.Field(nullable=True)
    # D28: pandera reads the annotation as the dtype, so a parametrized dtype instance is needed
    timestamp: UTC_MICROSECONDS  # type: ignore[valid-type]
    text: str
    is_bot: bool | None = pa.Field(nullable=True)

    class Config:
        """Reject columns outside the schema, so an author field cannot slip through."""

        strict = True

    @pa.dataframe_check
    @classmethod
    def parent_is_not_the_message(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """A message is never its own parent."""
        return data.lazyframe.select(_parent_differs())


class MessagesSchema(pa.DataFrameModel):
    """The messages boundary of D28: one row per kept message, with its duplicate group."""

    message_id: str = pa.Field(unique=True, str_length=NON_EMPTY)
    source: str = pa.Field(str_length=NON_EMPTY)
    platform: str = pa.Field(str_length=NON_EMPTY)
    channel_id: str = pa.Field(str_length=NON_EMPTY)
    parent_message_id: str | None = pa.Field(nullable=True)
    # D28: pandera reads the annotation as the dtype, so a parametrized dtype instance is needed
    timestamp: UTC_MICROSECONDS  # type: ignore[valid-type]
    text: str = pa.Field(str_length=NON_EMPTY)
    language: str = pa.Field(isin=[language.value for language in Language])
    language_confidence: float = pa.Field(ge=0, le=1)
    is_mixed: bool
    stratum: int = pa.Field(ge=0)
    context_source: str = pa.Field(isin=[source.value for source in ContextSource])
    dup_group_id: str = pa.Field(str_length=NON_EMPTY)

    class Config:
        """Reject columns outside the schema."""

        strict = True

    @pa.dataframe_check
    @classmethod
    def parent_is_not_the_message(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """A message is never its own parent."""
        return data.lazyframe.select(_parent_differs())

    @pa.dataframe_check
    @classmethod
    def dup_group_id_is_the_smallest_member(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """The group id of a duplicate group is its smallest message id."""
        smallest = pl.col(Column.MESSAGE_ID).min().over(Column.DUP_GROUP_ID)
        return data.lazyframe.select((smallest == pl.col(Column.DUP_GROUP_ID)).alias("ok"))

    @pa.dataframe_check
    @classmethod
    def context_source_follows_the_parent_link(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """A message has reply-link context exactly when its parent is a row of the table."""
        parent_present = pl.col(Column.PARENT_MESSAGE_ID).is_in(pl.col(Column.MESSAGE_ID).implode())
        linked = pl.col(Column.CONTEXT_SOURCE) == ContextSource.REPLY_LINK
        return data.lazyframe.select((parent_present.fill_null(value=False) == linked).alias("ok"))


def validate_frame[FrameT: (pl.DataFrame, pl.LazyFrame)](
    schema: type[pa.DataFrameModel], frame: FrameT
) -> FrameT:
    """Validate `frame` against `schema` and return it.

    A lazy frame is checked at the schema level only (columns and dtypes), so nothing is
    collected; `validate_lazy_data` checks its values.

    Raises:
        MessageSchemaError: The frame breaks the schema. The message names the schema, the
            column, the check and the number of failing rows, never the failing values, because
            pandera's own message quotes rows and messages are not to reach logs (H12).
    """
    try:
        return schema.validate(frame)
    except pandera.errors.SchemaError as error:
        raise MessageSchemaError(_describe(schema, error)) from None


def validate_lazy_data(schema: type[pa.DataFrameModel], frame: pl.LazyFrame) -> None:
    """Validate the columns, dtypes and values of a lazy frame, which polars evaluates lazily.

    Raises:
        MessageSchemaError: As `validate_frame`.
    """
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        validate_frame(schema, frame)


def _describe(schema: type[pa.DataFrameModel], error: pandera.errors.SchemaError) -> str:
    cases = error.failure_cases
    failing = cases.height if isinstance(cases, pl.DataFrame) else 0
    column = error.column_name
    if column is None and error.reason_code in NAME_ONLY_REASONS and isinstance(cases, str):
        column = cases
    return (
        f"{schema.__name__}: column {column!r} failed check {error.check!s} "
        f"({error.reason_code.name}), {failing} failing row(s)"
    )


def _parent_differs() -> pl.Expr:
    parent = pl.col(Column.PARENT_MESSAGE_ID)
    return (parent.is_null() | (parent != pl.col(Column.MESSAGE_ID))).alias("ok")
