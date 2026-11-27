"""The `Embedder` protocol (D14)."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    """Turns texts into L2-normalized float32 vectors."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed `texts` into an array of shape `(len(texts), dimension)`, one unit row each."""
        ...
