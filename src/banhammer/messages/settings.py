"""The `messages` section of `params.yaml` (S2): source, language ID, dedup and sampling."""

from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import Field, model_validator

from banhammer.core.constants import COMMIT_HASH_PATTERN, Language
from banhammer.core.settings import SettingsModel
from banhammer.messages.constants import SOURCE_SUFFIXES, UNPINNED_REVISION


class SourceColumns(SettingsModel):
    """Names of the source columns the adapter reads; `None` marks a column the source lacks."""

    message_id: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    parent_message_id: str | None
    timestamp: str = Field(min_length=1)
    text: str = Field(min_length=1)
    is_bot: str | None


class SourceSettings(SettingsModel):
    """The dataset to ingest: Hub id, pinned commit, files to fetch and column names."""

    repo_id: str = Field(min_length=1)
    revision: str = Field(pattern=COMMIT_HASH_PATTERN)
    allow_patterns: tuple[str, ...] = Field(min_length=1)
    etag_timeout: float = Field(gt=0)
    columns: SourceColumns

    @model_validator(mode="after")
    def _require_a_pinned_revision(self) -> Self:
        if self.revision == UNPINNED_REVISION:
            message = (
                f"revision not pinned: {self.repo_id} still has the all-zero placeholder, "
                "set the dataset commit hash in params.yaml"
            )
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _require_supported_suffixes(self) -> Self:
        unsupported = [
            pattern
            for pattern in self.allow_patterns
            if PurePosixPath(pattern).suffix not in SOURCE_SUFFIXES
        ]
        if unsupported:
            message = f"allow_patterns must end in one of {SOURCE_SUFFIXES}: {unsupported}"
            raise ValueError(message)
        return self


class LanguageSettings(SettingsModel):
    """Language identification (D19): detector set, minimum length and the two confidences.

    `confidence` is c, the least confidence of the top language; `mixed_confidence` is c2, the
    confidence of the other of English and Spanish from which a message is tagged mixed.
    """

    detector_languages: tuple[str, ...] = Field(min_length=2)
    min_length: int = Field(gt=0)
    confidence: float = Field(gt=0, le=1)
    mixed_confidence: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _require_kept_languages(self) -> Self:
        missing = [code for code in Language if code not in self.detector_languages]
        if missing:
            message = f"detector_languages must include {', '.join(missing)}"
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _require_distinct_codes(self) -> Self:
        if len(set(self.detector_languages)) != len(self.detector_languages):
            message = "detector_languages must not repeat a code"
            raise ValueError(message)
        return self


class DedupSettings(SettingsModel):
    """Duplicate detection (D13, D14): MinHash threshold t1, translation threshold t2, search.

    `lsh_recall_weight` is the weight given to missed pairs, against `1 - lsh_recall_weight` for
    false candidates, when LSH bands and rows are chosen for `jaccard_threshold`; candidates are
    verified with the exact Jaccard, so a false candidate only costs time. `minhash_seed` seeds
    the hash permutations; `max_group_size_warning` is the group size above which a warning
    with the size is logged.
    """

    shingle_size: int = Field(gt=0)
    num_perm: int = Field(gt=0)
    minhash_seed: int = Field(ge=0)
    lsh_recall_weight: float = Field(gt=0, lt=1)
    jaccard_threshold: float = Field(gt=0, le=1)
    translation_threshold: float = Field(gt=-1, le=1)
    candidate_floor: float = Field(gt=-1, le=1)
    k: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    max_group_size_warning: int = Field(gt=0)

    @model_validator(mode="after")
    def _floor_not_above_threshold(self) -> Self:
        if self.candidate_floor > self.translation_threshold:
            message = "candidate_floor must not exceed translation_threshold"
            raise ValueError(message)
        return self


class CalibrationSettings(SettingsModel):
    """Size of the human calibration set for t2: the top pairs plus a band around t2."""

    top_count: int = Field(ge=0)
    band_count: int = Field(ge=0)
    band_half_width: float = Field(gt=0, lt=2)


class SamplingSettings(SettingsModel):
    """Sampling plan (proposal 4.6): cap, window length and number of channel-size strata."""

    max_messages: int = Field(gt=0)
    window_size: int = Field(gt=0)
    strata: int = Field(gt=0)

    @model_validator(mode="after")
    def _share_holds_a_window(self) -> Self:
        if self.max_messages // self.strata < self.window_size:
            message = (
                f"max_messages // strata ({self.max_messages // self.strata}) must be at least "
                f"window_size ({self.window_size}), or a stratum's share cannot hold one window"
            )
            raise ValueError(message)
        return self


class MessagesSettings(SettingsModel):
    """The `messages` section of `params.yaml`."""

    raw_dir: Path
    output_dir: Path
    source: SourceSettings
    language: LanguageSettings
    dedup: DedupSettings
    calibration: CalibrationSettings
    sampling: SamplingSettings
