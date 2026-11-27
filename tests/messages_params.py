from pathlib import Path

import yaml

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.messages import MESSAGES_SECTION, MessagesSettings

REPO_ROOT = Path(__file__).resolve().parent.parent
UNPINNED = "0" * 40
PINNED = "ab" * 20


def repository_document() -> dict[str, dict[str, object]]:
    document: dict[str, dict[str, object]] = yaml.safe_load(
        (REPO_ROOT / PARAMS_FILE_NAME).read_text(encoding="utf-8")
    )
    return document


def pinned_section() -> dict[str, object]:
    """The `messages` section of the repository params, with the placeholder revision pinned."""
    section = repository_document()[MESSAGES_SECTION]
    source = dict(section["source"])  # type: ignore[call-overload]
    if source["revision"] == UNPINNED:
        source["revision"] = PINNED
    return section | {"source": source}


def messages_settings() -> MessagesSettings:
    return MessagesSettings.model_validate(pinned_section())
