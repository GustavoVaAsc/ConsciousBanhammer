from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class FakeEmbedder:
    """Embedder double: looks each text up in a table of vectors and L2-normalizes it."""

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self._vectors = {
            text: np.asarray(vector, dtype=np.float32) for text, vector in vectors.items()
        }
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(tuple(texts))
        dimension = len(next(iter(self._vectors.values())))
        matrix = (
            np.stack([self._vectors[text] for text in texts])
            if texts
            else np.empty((0, dimension), np.float32)
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)
