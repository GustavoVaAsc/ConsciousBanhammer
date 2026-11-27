"""The `embedder` section of `params.yaml` (D14)."""

from pydantic import Field

from banhammer.core.constants import COMMIT_HASH_PATTERN
from banhammer.core.settings import SettingsModel


class EmbedderSettings(SettingsModel):
    """The `embedder` section of `params.yaml`: which model, at which commit, how to run it.

    `revision` is a full commit hash, never a branch or a tag, so the weights cannot move.
    """

    model: str = Field(min_length=1)
    revision: str = Field(pattern=COMMIT_HASH_PATTERN)
    batch_size: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)
