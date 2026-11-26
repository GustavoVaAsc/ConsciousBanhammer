"""Sentence embeddings (D14): the `Embedder` protocol and the `embedder` settings section.

The backend lives in `sentence_transformer_embedder`, so importing this module does not import
`sentence_transformers`.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
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


class Embedder(Protocol):
    """Turns texts into L2-normalized float32 vectors."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed `texts` into an array of shape `(len(texts), dimension)`, one unit row each."""
        ...
