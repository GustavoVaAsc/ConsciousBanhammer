"""Fixed facts of the messages subpackage: file names, column names, enums and patterns."""

import re
from enum import StrEnum

MESSAGES_SECTION = "messages"

RAW_MESSAGES_FILE_NAME = "raw_messages.parquet"
MESSAGES_FILE_NAME = "messages.parquet"
MANIFEST_FILE_NAME = "messages_manifest.json"
CALIBRATION_FILE_NAME = "translation_calibration.jsonl"
INGEST_REPORT_FILE_NAME = "ingest_report.json"

DISCORD_PLATFORM = "discord"
DISCORD_ID_PREFIX = "discord"
ID_SEPARATOR = ":"

SOURCE_SUFFIXES = (".parquet",)

UNPINNED_REVISION = "0" * 40
HUB_CACHE_DIRECTORY = ".cache"

# https://discord.com/developers/docs/resources/message: a message holds at most 4000 characters
MAX_TEXT_CHARS = 4000

MIN_DETECTOR_LANGUAGES = 2
BOT_FLAG_WORDS = {"true": True, "false": False, "1": True, "0": False}
GROUP_SIZE_BUCKET_BOUNDS = (1, 2, 4, 8, 16)

SAMPLING_STREAM = "messages.sampling"
CALIBRATION_STREAM = "messages.calibration"


class Column(StrEnum):
    """Column names of the raw-message and messages frames."""

    MESSAGE_ID = "message_id"
    SOURCE = "source"
    PLATFORM = "platform"
    CHANNEL_ID = "channel_id"
    PARENT_MESSAGE_ID = "parent_message_id"
    TIMESTAMP = "timestamp"
    TEXT = "text"
    IS_BOT = "is_bot"
    LANGUAGE = "language"
    LANGUAGE_CONFIDENCE = "language_confidence"
    IS_MIXED = "is_mixed"
    STRATUM = "stratum"
    CONTEXT_SOURCE = "context_source"
    DUP_GROUP_ID = "dup_group_id"


class ContextSource(StrEnum):
    """Where a message's context comes from (D16)."""

    REPLY_LINK = "reply_link"
    NONE = "none"


class DropReason(StrEnum):
    """Why language identification dropped a message."""

    TOO_SHORT = "too_short"
    OTHER_LANGUAGE = "other_language"
    LOW_CONFIDENCE = "low_confidence"


class CalibrationOrigin(StrEnum):
    """Which part of the calibration candidate set a pair was drawn into."""

    TOP = "top"
    BAND = "band"


class PairColumn(StrEnum):
    """Columns of the translation candidate pairs and of the calibration export."""

    EN_MESSAGE_ID = "en_message_id"
    ES_MESSAGE_ID = "es_message_id"
    EN_TEXT = "en_text"
    ES_TEXT = "es_text"
    COSINE = "cosine"
    ORIGIN = "origin"
    IS_TRANSLATION = "is_translation"


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
USER_MENTION_PATTERN = re.compile(r"<@!?\d+>")
ROLE_MENTION_PATTERN = re.compile(r"<@&\d+>")
CHANNEL_MENTION_PATTERN = re.compile(r"<#\d+>")
USER_PLACEHOLDER = "<@user>"
ROLE_PLACEHOLDER = "<@role>"
CHANNEL_PLACEHOLDER = "<#channel>"
DISCORD_LINK_PATTERN = re.compile(
    r"(?i)(?:https?://)?(?:(?:ptb|canary)\.)?discord(?:app)?\.com/(?:users/\d+|channels/\d+(?:/\d+){1,2})"
)
DISCORD_LINK_PLACEHOLDER = "<discord-link>"
MENTION_REPLACEMENTS = (
    (DISCORD_LINK_PATTERN, DISCORD_LINK_PLACEHOLDER),
    (USER_MENTION_PATTERN, USER_PLACEHOLDER),
    (ROLE_MENTION_PATTERN, ROLE_PLACEHOLDER),
    (CHANNEL_MENTION_PATTERN, CHANNEL_PLACEHOLDER),
)
MENTION_PATTERN = re.compile(
    r"<(?:@[!&]?\d+|#\d+)>|<@user>|<@role>|<#channel>|<discord-link>|(?<![\w@])@\w+"
)
CUSTOM_EMOJI_PATTERN = re.compile(r"<a?:\w+:\d+>|(?<!\w):[a-z0-9_+-]{2,}:(?!\w)", re.IGNORECASE)
CODE_BLOCK_PATTERN = re.compile(r"```[^`]*(?:`(?!``)[^`]*)*```|`[^`\n]*`")
WHITESPACE_PATTERN = re.compile(r"\s+")
EMOJI_CATEGORY = "So"
EMOJI_SKIN_TONES = range(0x1F3FB, 0x1F400)
EMOJI_JOINERS = frozenset({"\u200d", "\ufe0f", "\ufe0e", "\u20e3"})
