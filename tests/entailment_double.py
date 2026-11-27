from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class FakeEntailmentScorer:
    """EntailmentScorer double: looks each (premise, hypothesis) up in a table of probabilities."""

    def __init__(
        self, probabilities: Mapping[tuple[str, str], float], default: float = 0.0
    ) -> None:
        self._probabilities = dict(probabilities)
        self._default = default
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def entailment_probabilities(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> NDArray[np.float32]:
        self.calls.append((tuple(premises), tuple(hypotheses)))
        return np.array(
            [
                self._probabilities.get(pair, self._default)
                for pair in zip(premises, hypotheses, strict=True)
            ],
            dtype=np.float32,
        )
