"""Exceptions of the messages subpackage."""

from banhammer.core.errors import BanhammerError


class MessagesError(BanhammerError):
    """Base class of the exceptions of the messages subpackage."""


class SourceFormatError(MessagesError):
    """A source file is missing, has an unsupported format, or lacks an expected column."""


class SourceDownloadError(MessagesError):
    """The source dataset could not be downloaded at its pinned revision."""


class MessageSchemaError(MessagesError):
    """A frame does not satisfy its boundary schema; the message names checks, never data."""


class LanguageDetectionError(MessagesError):
    """The language detector cannot be built from the configured languages."""


class SamplingError(MessagesError):
    """The sampler received a frame or parameters it cannot work with."""


class DeduplicationError(MessagesError):
    """The duplicate search received input it cannot work with."""
