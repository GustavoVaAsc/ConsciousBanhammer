from collections.abc import Mapping

import polars as pl
import pytest

from banhammer.messages import (
    DropReason,
    LanguageDetectionError,
    LanguageFilter,
    LanguageSettings,
    LinguaLanguageDetector,
)
from message_frames import raw_frame

SETTINGS = LanguageSettings(
    detector_languages=("en", "es", "pt", "fr"),
    min_length=10,
    confidence=0.6,
    mixed_confidence=0.2,
)
CODES = ("en", "es", "pt", "fr")
ENGLISH = "this is clearly english"
SPANISH = "esto es claramente español"


class FakeDetector:
    def __init__(self, table: Mapping[str, Mapping[str, float]]) -> None:
        self._table = table
        self.seen: list[str] = []

    def confidences(self, text: str) -> Mapping[str, float]:
        self.seen.append(text)
        return dict.fromkeys(CODES, 0.0) | dict(self._table.get(text, {}))


def classify(
    table: Mapping[str, Mapping[str, float]], texts: list[str]
) -> tuple[pl.DataFrame, Mapping[DropReason, int], FakeDetector]:
    detector = FakeDetector(table)
    frame = raw_frame([{"message_id": f"m{i}", "text": text} for i, text in enumerate(texts)])
    result = LanguageFilter(detector, SETTINGS).annotate(frame)
    return result.kept, result.dropped, detector


def test_english_and_spanish_are_kept_with_language_and_confidence() -> None:
    table = {ENGLISH: {"en": 0.9, "es": 0.05}, SPANISH: {"es": 0.8, "pt": 0.15}}
    kept, dropped, _ = classify(table, [ENGLISH, SPANISH])
    assert kept["language"].to_list() == ["en", "es"]
    assert kept["language_confidence"].to_list() == [0.9, 0.8]
    assert kept["is_mixed"].to_list() == [False, False]
    assert all(count == 0 for count in dropped.values())


def test_confidence_exactly_c_is_kept_and_below_c_is_dropped() -> None:
    table = {ENGLISH: {"en": 0.6, "fr": 0.4}, SPANISH: {"es": 0.59, "pt": 0.41}}
    kept, dropped, _ = classify(table, [ENGLISH, SPANISH])
    assert kept["message_id"].to_list() == ["m0"]
    assert dropped[DropReason.LOW_CONFIDENCE] == 1


def test_the_other_language_at_least_c2_tags_the_message_mixed() -> None:
    table = {
        ENGLISH: {"en": 0.7, "es": 0.2, "fr": 0.1},
        SPANISH: {"es": 0.75, "en": 0.19, "pt": 0.06},
    }
    kept, _, _ = classify(table, [ENGLISH, SPANISH])
    assert kept["language"].to_list() == ["en", "es"]
    assert kept["is_mixed"].to_list() == [True, False]


def test_another_top_language_or_no_signal_is_dropped() -> None:
    table = {ENGLISH: {"pt": 0.9, "en": 0.05}, SPANISH: {}}
    kept, dropped, _ = classify(table, [ENGLISH, SPANISH])
    assert kept.height == 0
    assert dropped[DropReason.OTHER_LANGUAGE] == 1
    assert dropped[DropReason.LOW_CONFIDENCE] == 1


def test_length_is_measured_after_cleaning_and_the_detector_sees_cleaned_text() -> None:
    exactly = "a" * 10
    almost = "a" * 9
    noisy = "https://example.com/long/link <@123> 😂😂😂"
    table = {exactly: {"en": 0.9}, almost: {"en": 0.9}}
    kept, dropped, detector = classify(table, [exactly, almost, noisy, f"{exactly} <@1> 😂"])
    assert kept["message_id"].to_list() == ["m0", "m3"]
    assert dropped[DropReason.TOO_SHORT] == 2
    assert detector.seen == [exactly, exactly]


def test_kept_rows_keep_their_columns_and_order() -> None:
    table = {ENGLISH: {"en": 0.9}, SPANISH: {"es": 0.9}}
    frame = raw_frame(
        [
            {"message_id": "m2", "text": SPANISH, "channel_id": "c9"},
            {"message_id": "m1", "text": "short"},
            {"message_id": "m3", "text": ENGLISH},
        ]
    )
    kept = LanguageFilter(FakeDetector(table), SETTINGS).annotate(frame).kept
    assert kept["message_id"].to_list() == ["m2", "m3"]
    assert kept["channel_id"].to_list() == ["c9", "discord:c1"]
    assert kept.columns == [*frame.columns, "language", "language_confidence", "is_mixed"]
    assert kept["text"].to_list() == [SPANISH, ENGLISH]


def test_an_empty_frame_gives_an_empty_result() -> None:
    kept, dropped, _ = classify({}, [])
    assert kept.height == 0
    assert kept.columns[-3:] == ["language", "language_confidence", "is_mixed"]
    assert sum(dropped.values()) == 0


def test_the_real_detector_separates_english_spanish_and_others() -> None:
    detector = LinguaLanguageDetector(SETTINGS.detector_languages)
    frame = raw_frame(
        [
            {"message_id": "en", "text": "hello how are you doing today my friend"},
            {"message_id": "es", "text": "hola cómo estás hoy amigo mío"},
            {"message_id": "fr", "text": "Bonjour tout le monde, comment ça va"},
            {"message_id": "pt", "text": "Olá pessoal, tudo bem com vocês hoje"},
            {"message_id": "url", "text": "https://example.com/some/page"},
        ]
    )
    result = LanguageFilter(detector, SETTINGS).annotate(frame)
    assert dict(zip(result.kept["message_id"], result.kept["language"], strict=True)) == {
        "en": "en",
        "es": "es",
    }
    assert result.dropped[DropReason.OTHER_LANGUAGE] == 2
    assert result.dropped[DropReason.TOO_SHORT] == 1


def test_the_real_detector_reports_a_confidence_per_configured_language() -> None:
    detector = LinguaLanguageDetector(CODES)
    confidences = detector.confidences("this is a perfectly ordinary english sentence")
    assert set(confidences) == set(CODES)
    assert sum(confidences.values()) == pytest.approx(1.0)
    assert max(confidences, key=confidences.__getitem__) == "en"


def test_the_real_detector_reports_zeros_for_text_without_letters() -> None:
    assert set(LinguaLanguageDetector(CODES).confidences("12345 !!!").values()) == {0.0}


@pytest.mark.parametrize("codes", [("en", "gl"), ("en", "xx"), ("en",), ()])
def test_the_real_detector_rejects_unsupported_codes_and_too_few_languages(
    codes: tuple[str, ...],
) -> None:
    with pytest.raises(LanguageDetectionError):
        LinguaLanguageDetector(codes)
