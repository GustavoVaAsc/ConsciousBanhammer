"""Language identification (D19): clean, detect, keep English and Spanish, tag mixed."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import polars as pl
from lingua import IsoCode639_1, LanguageDetectorBuilder

from banhammer.core.constants import Language
from banhammer.messages.constants import MIN_DETECTOR_LANGUAGES, Column, DropReason
from banhammer.messages.errors import LanguageDetectionError
from banhammer.messages.settings import LanguageSettings
from banhammer.messages.text_cleaning import clean_text


class LanguageDetector(Protocol):
    """Gives the confidence of each language of a fixed set for a text."""

    def confidences(self, text: str) -> Mapping[str, float]:
        """Confidence per ISO 639-1 code, all zero when the text has no usable signal."""
        ...


class LinguaLanguageDetector:
    """`lingua-py` restricted to a set of languages (D19); confidences sum to 1 over the set."""

    def __init__(self, languages: Sequence[str]) -> None:
        """Build the detector for ISO 639-1 codes.

        Raises:
            LanguageDetectionError: Fewer than two codes, or a code lingua does not support
                (Galician, `gl`, is not among them).
        """
        resolved = {
            language: getattr(IsoCode639_1, language.upper(), None) for language in languages
        }
        unsupported = [language for language, code in resolved.items() if code is None]
        if unsupported or len(resolved) < MIN_DETECTOR_LANGUAGES:
            message = (
                f"cannot build a detector from {list(languages)}: need at least "
                f"{MIN_DETECTOR_LANGUAGES} languages, unsupported: {unsupported}"
            )
            raise LanguageDetectionError(message)
        codes = [code for code in resolved.values() if code is not None]
        self._detector = LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()

    def confidences(self, text: str) -> Mapping[str, float]:
        """Confidence per configured language."""
        return {
            value.language.iso_code_639_1.name.lower(): float(value.value)
            for value in self._detector.compute_language_confidence_values(text)
        }


@dataclass(frozen=True)
class LanguageResult:
    """Outcome of language identification: the kept messages and how many were dropped why.

    `kept` has the input columns plus `language`, `language_confidence` and `is_mixed`;
    `dropped` has an entry for every `DropReason`.
    """

    kept: pl.DataFrame
    dropped: Mapping[DropReason, int]


@dataclass(frozen=True)
class _Verdict:
    language: str | None
    confidence: float
    is_mixed: bool
    drop: DropReason | None


class LanguageFilter:
    """Keeps messages whose top language is English or Spanish with enough confidence (D19).

    Text is cleaned (URLs, mentions, emojis and code removed) before its length is measured and
    before detection. A kept message is tagged mixed when the other of English and Spanish
    reaches `mixed_confidence`.
    """

    def __init__(self, detector: LanguageDetector, settings: LanguageSettings) -> None:
        """Take the detector (a double in tests) and the language settings."""
        self._detector = detector
        self._settings = settings

    def annotate(self, frame: pl.DataFrame) -> LanguageResult:
        """Identify the language of every row of `frame`, which needs a `text` column."""
        verdicts = [self._classify(clean_text(text)) for text in frame[Column.TEXT]]
        annotated = frame.with_columns(
            pl.Series(Column.LANGUAGE, [v.language for v in verdicts], dtype=pl.String),
            pl.Series(Column.LANGUAGE_CONFIDENCE, [v.confidence for v in verdicts], pl.Float64),
            pl.Series(Column.IS_MIXED, [v.is_mixed for v in verdicts], dtype=pl.Boolean),
        )
        counts = Counter(v.drop for v in verdicts)
        return LanguageResult(
            kept=annotated.filter(pl.col(Column.LANGUAGE).is_not_null()),
            dropped={reason: counts[reason] for reason in DropReason},
        )

    def _classify(self, text: str) -> _Verdict:
        if len(text) < self._settings.min_length:
            return _dropped(DropReason.TOO_SHORT)
        confidences = self._detector.confidences(text)
        top = max(sorted(confidences), key=confidences.__getitem__)
        if confidences[top] <= 0:
            return _dropped(DropReason.LOW_CONFIDENCE)
        if top not in set(Language):
            return _dropped(DropReason.OTHER_LANGUAGE)
        if confidences[top] < self._settings.confidence:
            return _dropped(DropReason.LOW_CONFIDENCE)
        other = Language.ES if top == Language.EN else Language.EN
        is_mixed = confidences.get(other, 0.0) >= self._settings.mixed_confidence
        return _Verdict(top, confidences[top], is_mixed, None)


def _dropped(reason: DropReason) -> _Verdict:
    return _Verdict(None, 0.0, is_mixed=False, drop=reason)
