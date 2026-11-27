"""Keeps polars error text, which can quote data values, out of exceptions.

Functional module: one stateless context manager.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import polars as pl

from banhammer.messages.errors import SourceFormatError


@contextmanager
def hidden_polars_errors() -> Iterator[None]:
    """Re-raise any polars error as `SourceFormatError` carrying only the error class name.

    Polars messages can quote the offending value (a parse failure, a dtype inference), and
    source values must not reach logs (H12). The original error is not chained.

    Raises:
        SourceFormatError: A polars error was raised inside the block.
    """
    try:
        yield
    except pl.exceptions.PolarsError as error:
        raise SourceFormatError(type(error).__name__) from None
