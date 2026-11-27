"""The `EntailmentScorer` protocol: how likely one rule text entails another (D15)."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EntailmentScorer(Protocol):
    """Scores natural-language-inference entailment between texts, possibly across languages."""

    def entailment_probabilities(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> NDArray[np.float32]:
        """Probability that `premises[i]` entails `hypotheses[i]`, as an array of shape `(n,)`."""
        ...
