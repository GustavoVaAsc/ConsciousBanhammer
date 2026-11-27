import numpy as np
import polars as pl
import pytest

from banhammer.core.seeding import Seeder
from banhammer.messages import CalibrationSetBuilder, CalibrationSettings, DeduplicationError
from banhammer.messages.constants import CALIBRATION_STREAM

THRESHOLD = 0.85
COLUMNS = [
    "en_message_id",
    "es_message_id",
    "en_text",
    "es_text",
    "cosine",
    "origin",
    "is_translation",
]


def builder(
    top: int = 10, band: int = 10, half_width: float = 0.05, seed: int = 0
) -> CalibrationSetBuilder:
    settings = CalibrationSettings(top_count=top, band_count=band, band_half_width=half_width)
    return CalibrationSetBuilder(
        settings, THRESHOLD, Seeder(seed).numpy_generator(CALIBRATION_STREAM)
    )


def pairs(count: int = 200) -> tuple[pl.DataFrame, pl.DataFrame]:
    cosines = np.linspace(1.0, 0.70, count)
    candidates = pl.DataFrame(
        {
            "en_message_id": [f"e{i:03d}" for i in range(count)],
            "es_message_id": [f"s{i:03d}" for i in range(count)],
            "cosine": cosines,
        }
    )
    messages = pl.DataFrame(
        {
            "message_id": [f"{side}{i:03d}" for i in range(count) for side in "es"],
            "text": [f"{side}-text-{i:03d}" for i in range(count) for side in "es"],
        }
    )
    return candidates, messages


def test_the_set_holds_the_top_pairs_and_a_band_around_the_threshold() -> None:
    candidates, messages = pairs()
    result = builder(top=10, band=15).build(candidates, messages)
    assert result.columns == COLUMNS
    top = result.filter(pl.col("origin") == "top")
    band = result.filter(pl.col("origin") == "band")
    assert sorted(top["en_message_id"]) == [f"e{i:03d}" for i in range(10)]
    assert band.height == 15
    assert band["cosine"].is_between(THRESHOLD - 0.05, THRESHOLD + 0.05).all()
    assert set(top["en_message_id"]).isdisjoint(band["en_message_id"])


def test_texts_are_looked_up_by_message_id() -> None:
    candidates, messages = pairs()
    result = builder().build(candidates, messages)
    for row in result.iter_rows(named=True):
        assert row["en_text"] == f"e-text-{row['en_message_id'][1:]}"
        assert row["es_text"] == f"s-text-{row['es_message_id'][1:]}"


def test_the_label_column_is_empty_for_the_human_to_fill() -> None:
    result = builder().build(*pairs())
    assert result["is_translation"].null_count() == result.height


def test_rows_are_shuffled_so_the_score_order_does_not_guide_the_labeller() -> None:
    result = builder(top=20, band=0).build(*pairs())
    assert result["cosine"].to_list() != sorted(result["cosine"], reverse=True)


def test_the_same_seed_gives_the_same_set_and_another_seed_a_different_band() -> None:
    first = builder(seed=1).build(*pairs())
    assert first.equals(builder(seed=1).build(*pairs()))
    assert not first.equals(builder(seed=2).build(*pairs()))


def test_fewer_pairs_than_requested_gives_what_exists() -> None:
    candidates, messages = pairs(count=12)
    result = builder(top=5, band=10).build(candidates, messages)
    assert result["origin"].value_counts().sort("origin").rows() == [("band", 3), ("top", 5)]
    assert result["en_message_id"].is_unique().all()


def test_no_candidates_gives_an_empty_set_with_the_columns() -> None:
    candidates, messages = pairs()
    result = builder().build(candidates.clear(), messages)
    assert result.height == 0
    assert result.columns == COLUMNS


def test_a_pair_without_its_message_text_is_an_error() -> None:
    candidates, messages = pairs()
    with pytest.raises(DeduplicationError, match="text"):
        builder().build(candidates, messages.filter(pl.col("message_id") != "e000"))
