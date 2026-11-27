"""Distances between rule embeddings.

Functional module: stateless metric formulas over embedding matrices.
"""

import numpy as np
from numpy.typing import NDArray

from banhammer.rules.constants import COSINE_DISTANCE_MAX, MIN_MEMBERS_FOR_A_PAIR


def cluster_diameter(vectors: NDArray[np.float32]) -> float:
    """Largest pairwise cosine distance between the rows of an L2-normalized matrix.

    The distance between two unit rows is one minus their dot product, so it lies in `[0, 2]`.
    Fewer than two rows have diameter 0. A cluster is only as tight as its two furthest
    members, which is what exposes chaining: A close to B and B close to C do not make A close
    to C.
    """
    if len(vectors) < MIN_MEMBERS_FOR_A_PAIR:
        return 0.0
    rows = vectors.astype(np.float64)
    distances = 1.0 - rows @ rows.T
    return float(np.clip(distances.max(), 0.0, COSINE_DISTANCE_MAX))
