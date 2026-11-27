"""The `rules` section of `params.yaml`: directories, candidate search, thresholds, NLI model."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from banhammer.core.constants import COMMIT_HASH_PATTERN
from banhammer.core.settings import SettingsModel
from banhammer.rules.constants import (
    COSINE_DISTANCE_MAX,
    COSINE_MAX,
    COSINE_MIN,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
)


class NliSettings(SettingsModel):
    """The multilingual NLI model behind the mutual-entailment test (D15, D26).

    `revision` is a full commit hash, never a branch or a tag, so the weights cannot move.
    """

    model: str = Field(min_length=1)
    revision: str = Field(pattern=COMMIT_HASH_PATTERN)
    batch_size: int = Field(gt=0)
    max_length: int = Field(gt=0)


class RulesSettings(SettingsModel):
    """The `rules` section of `params.yaml`.

    Candidates are the `neighbours_k` nearest items with cosine of at least
    `neighbours_min_cosine`. A pair is an edge when the entailment probability is at least
    `entail_high` both ways, no edge when it is at most `entail_low` both ways, and goes to
    manual review otherwise. A cluster whose largest pairwise cosine distance exceeds
    `diameter_max` goes to split review. `saturation_step` is the number of rulesets between
    two points of the saturation curve.
    """

    rulesets_dir: Path
    output_dir: Path
    review_dir: Path
    neighbours_k: int = Field(gt=0)
    neighbours_min_cosine: float = Field(ge=COSINE_MIN, le=COSINE_MAX)
    neighbours_chunk_size: int = Field(gt=0)
    entail_high: float = Field(ge=PROBABILITY_MIN, le=PROBABILITY_MAX)
    entail_low: float = Field(ge=PROBABILITY_MIN, le=PROBABILITY_MAX)
    diameter_max: float = Field(gt=0, le=COSINE_DISTANCE_MAX)
    saturation_step: int = Field(gt=0)
    nli: NliSettings

    @model_validator(mode="after")
    def _low_threshold_below_high(self) -> Self:
        if self.entail_low >= self.entail_high:
            message = (
                f"entail_low ({self.entail_low}) must be below entail_high ({self.entail_high})"
            )
            raise ValueError(message)
        return self
