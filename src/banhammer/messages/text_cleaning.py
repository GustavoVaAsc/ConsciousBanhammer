"""Text cleaning for language identification, embedding and duplicate keys.

Functional module: stateless text transformations.
"""

import unicodedata

from banhammer.messages.constants import (
    CODE_BLOCK_PATTERN,
    CUSTOM_EMOJI_PATTERN,
    EMOJI_CATEGORY,
    EMOJI_JOINERS,
    EMOJI_SKIN_TONES,
    MENTION_PATTERN,
    URL_PATTERN,
    WHITESPACE_PATTERN,
)

NOISE_PATTERNS = (CODE_BLOCK_PATTERN, URL_PATTERN, MENTION_PATTERN, CUSTOM_EMOJI_PATTERN)


def clean_text(text: str) -> str:
    """Strip code blocks, URLs, mentions and emojis, apply NFKC and collapse whitespace.

    What is left is what language identification, the embedder and the duplicate key see.
    """
    cleaned = unicodedata.normalize("NFKC", text)
    for pattern in NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    without_emoji = "".join(char for char in cleaned if not _is_emoji(char))
    return WHITESPACE_PATTERN.sub(" ", without_emoji).strip()


def dedup_key(text: str) -> str:
    """The cleaned text, case-folded: equal keys mean the texts differ only in noise and case."""
    return clean_text(text).casefold()


def _is_emoji(char: str) -> bool:
    return (
        char in EMOJI_JOINERS
        or ord(char) in EMOJI_SKIN_TONES
        or unicodedata.category(char) == EMOJI_CATEGORY
    )
