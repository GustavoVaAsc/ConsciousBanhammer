"""`EntailmentScorer` backed by a multilingual NLI cross-encoder, loaded from a snapshot (D26)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import CrossEncoder

from banhammer.core.model_snapshot import download_snapshot, verify_snapshot
from banhammer.rules.constants import ENTAILMENT_LABEL
from banhammer.rules.errors import EntailmentError
from banhammer.rules.settings import NliSettings


class CrossEncoderEntailmentScorer:
    """Scores entailment with a sequence-classification NLI model run as a cross-encoder.

    The model loads offline, from safetensors, without remote code (H16). The column of the
    entailment class is found by its label name in the model configuration, not by position.
    """

    def __init__(
        self, settings: NliSettings, device: torch.device, dtype: torch.dtype, snapshot_dir: Path
    ) -> None:
        """Load the model stored in `snapshot_dir` onto `device` in `dtype`.

        Raises:
            ModelSnapshotError: `snapshot_dir` holds a file outside the allowed patterns.
            EntailmentError: The model does not have exactly one entailment label.
        """
        verify_snapshot(snapshot_dir)
        self._settings = settings
        self._model = CrossEncoder(
            str(snapshot_dir),
            device=str(device),
            trust_remote_code=False,
            local_files_only=True,
            max_length=settings.max_length,
            model_kwargs={"use_safetensors": True, "dtype": dtype},
        )
        self._entailment_column = self._find_entailment_column()

    @classmethod
    def from_hub(cls, settings: NliSettings, device: torch.device, dtype: torch.dtype) -> Self:
        """Download the pinned snapshot named in `settings` and load it onto `device`."""
        return cls(settings, device, dtype, download_snapshot(settings.model, settings.revision))

    def entailment_probabilities(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> NDArray[np.float32]:
        """Probability that each premise entails the hypothesis at the same position.

        Raises:
            EntailmentError: The two sequences differ in length.
        """
        if len(premises) != len(hypotheses):
            message = f"{len(premises)} premises but {len(hypotheses)} hypotheses"
            raise EntailmentError(message)
        if not premises:
            return np.empty(0, dtype=np.float32)
        with torch.inference_mode():
            probabilities = self._model.predict(
                list(zip(premises, hypotheses, strict=True)),
                batch_size=self._settings.batch_size,
                apply_softmax=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(probabilities[:, self._entailment_column], dtype=np.float32)

    def _find_entailment_column(self) -> int:
        columns: list[int] = [
            index
            for index, label in self._model.config.id2label.items()
            if label.lower() == ENTAILMENT_LABEL
        ]
        if len(columns) != 1:
            message = (
                f"{self._settings.model} must have exactly one {ENTAILMENT_LABEL!r} label, "
                f"found {self._model.config.id2label}"
            )
            raise EntailmentError(message)
        return columns[0]
