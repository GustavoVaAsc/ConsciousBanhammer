"""`Embedder` backed by a local `sentence-transformers` snapshot (D14)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from banhammer.core.embedding import EmbedderSettings
from banhammer.core.errors import EmbeddingError
from banhammer.core.model_snapshot import download_snapshot


class SentenceTransformerEmbedder:
    """Embeds texts with a sentence-transformers model loaded from a local snapshot directory.

    The load is offline, from safetensors, without remote code (H16).
    """

    def __init__(
        self, settings: EmbedderSettings, device: torch.device, snapshot_dir: Path
    ) -> None:
        """Load the model stored in `snapshot_dir` onto `device`."""
        self._settings = settings
        self._model = SentenceTransformer(
            str(snapshot_dir),
            device=str(device),
            trust_remote_code=False,
            local_files_only=True,
            model_kwargs={"use_safetensors": True},
        )
        self._model.max_seq_length = settings.max_seq_length

    @classmethod
    def from_hub(cls, settings: EmbedderSettings, device: torch.device) -> Self:
        """Download the pinned snapshot named in `settings` and load it onto `device`."""
        return cls(settings, device, download_snapshot(settings.model, settings.revision))

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed `texts` into an array of shape `(len(texts), dimension)`, one unit row each.

        Raises:
            EmbeddingError: `texts` is empty and the model does not report its dimension.
        """
        if not texts:
            return np.empty((0, self._dimension()), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self._settings.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _dimension(self) -> int:
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            message = f"{self._settings.model} does not report its embedding dimension"
            raise EmbeddingError(message)
        return dimension
