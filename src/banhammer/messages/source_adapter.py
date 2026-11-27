"""The contract between a message source and the rest of S2."""

from typing import Protocol

import polars as pl

from banhammer.messages.manifest import IngestReport


class SourceAdapter(Protocol):
    """Reads one message source into the raw-message schema.

    Implementations map source columns to `RawMessageSchema`, namespace every id with a source
    prefix so ids stay unique across sources, convert timestamps to UTC microseconds, drop
    author fields (only the columns of `RawMessageSchema` are delivered, proposal 4.6) and
    pseudonymize anything in the text that names a user.
    """

    def scan(self) -> pl.LazyFrame:
        """The source's messages as a lazy frame with exactly the `RawMessageSchema` columns."""
        ...

    def report(self) -> IngestReport:
        """Count what `scan` drops or rewrites, and fail on values it cannot parse."""
        ...
