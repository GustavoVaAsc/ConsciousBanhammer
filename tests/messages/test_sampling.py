import logging
from collections.abc import Callable, Mapping

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from banhammer.core.seeding import Seeder
from banhammer.messages import (
    ChannelSampler,
    DropReason,
    LanguageFilter,
    LanguageSettings,
    SamplingError,
    SamplingSettings,
)
from banhammer.messages.constants import SAMPLING_STREAM
from message_frames import EPOCH, raw_frame

ENGLISH = "hello there everyone"
FRENCH = "bonjour tout le monde ici"
LANGUAGE = LanguageSettings(
    detector_languages=("en", "es"), min_length=10, confidence=0.6, mixed_confidence=0.2
)


class SpyDetector:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def confidences(self, text: str) -> Mapping[str, float]:
        self.seen.append(text)
        if text == FRENCH:
            return {"en": 0.05, "es": 0.05, "fr": 0.9}
        return {"en": 0.9, "es": 0.05, "fr": 0.05}


def sampler(
    max_messages: int = 1000,
    window_size: int = 5,
    strata: int = 3,
    seed: int = 0,
    detector: SpyDetector | None = None,
) -> ChannelSampler:
    config = SamplingSettings(max_messages=max_messages, window_size=window_size, strata=strata)
    language = LanguageFilter(detector or SpyDetector(), LANGUAGE)
    return ChannelSampler(config, Seeder(seed).numpy_generator(SAMPLING_STREAM), language)


def pool(
    channel_sizes: dict[str, int], french: Callable[[str, int], bool] = lambda *_: False
) -> pl.LazyFrame:
    rows = [
        {
            "message_id": f"{channel}-{index:04d}",
            "channel_id": channel,
            "text": FRENCH if french(channel, index) else ENGLISH,
        }
        for channel, size in channel_sizes.items()
        for index in range(size)
    ]
    frame = raw_frame(rows).drop("is_bot")
    offsets = pl.int_range(pl.len()).over("channel_id")
    return frame.with_columns(
        (pl.lit(EPOCH) + pl.duration(seconds=offsets)).dt.cast_time_unit("us").alias("timestamp")
    ).lazy()


CHANNELS = {f"c{k:02d}": size for k, size in enumerate([3, 8, 8, 12, 20, 33, 47, 50, 90, 120])}


def ids(result_frame: pl.DataFrame) -> list[str]:
    return result_frame["message_id"].to_list()


def test_the_sample_never_exceeds_the_cap_and_is_a_subset_of_the_pool() -> None:
    frame = pool(CHANNELS)
    result = sampler(max_messages=100, window_size=5, strata=5).sample(frame)
    assert result.sampled.height <= 100
    assert set(ids(result.sampled)) <= set(ids(frame.collect()))
    assert result.sampled["message_id"].is_unique().all()


def test_a_cap_above_the_pool_keeps_every_message_with_language_columns() -> None:
    frame = pool(CHANNELS).collect()
    result = sampler(max_messages=10_000).sample(frame.lazy())
    assert sorted(ids(result.sampled)) == sorted(ids(frame))
    assert {"language", "language_confidence", "is_mixed", "stratum"} <= set(result.sampled.columns)


def test_strata_are_deciles_of_channel_size_with_equal_channel_counts() -> None:
    result = sampler(max_messages=10_000, strata=5).sample(pool(CHANNELS))
    stratum_of = dict(
        result.sampled.group_by("channel_id").agg(pl.col("stratum").first()).iter_rows()
    )
    by_size = sorted(CHANNELS, key=lambda channel: (CHANNELS[channel], channel))
    assert [stratum_of[channel] for channel in by_size] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert result.eligible_per_stratum == {0: 11, 1: 20, 2: 53, 3: 97, 4: 210}


def test_strata_share_the_budget_equally_instead_of_following_pool_size() -> None:
    result = sampler(max_messages=100, window_size=1, strata=5).sample(pool(CHANNELS))
    assert result.sampled_per_stratum == {0: 11, 1: 20, 2: 23, 3: 23, 4: 23}
    assert sum(result.sampled_per_stratum.values()) == 100


def test_messages_are_taken_in_whole_contiguous_windows_of_a_channel() -> None:
    result = sampler(max_messages=40, window_size=5, strata=1).sample(
        pool({"big": 60, "other": 60})
    )
    for _, group in result.sampled.group_by("channel_id"):
        positions = sorted(int(identifier.split("-")[1]) for identifier in group["message_id"])
        windows = [positions[i : i + 5] for i in range(0, len(positions), 5)]
        assert all(window == list(range(window[0], window[0] + 5)) for window in windows)
        assert all(window[0] % 5 == 0 for window in windows)


def test_windows_follow_the_candidate_order_not_the_raw_order() -> None:
    frame = pool({"c": 10}).filter(pl.col("message_id") != "c-0002")
    result = sampler(max_messages=100, window_size=3, strata=1).sample(frame)
    assert result.sampled.height == 9


def test_windows_are_cut_before_the_language_filter_so_it_never_shifts_a_boundary() -> None:
    frame = pool({"c": 10}, french=lambda _, index: index in {2, 3})
    first_window = {"c-0000", "c-0001", "c-0004"}
    second_window = {f"c-{index:04d}" for index in range(5, 10)}
    outcomes = {
        frozenset(
            ids(sampler(max_messages=5, window_size=5, strata=1, seed=seed).sample(frame).sampled)
        )
        for seed in range(20)
    }
    assert outcomes == {frozenset(first_window), frozenset(second_window)}


def test_the_cap_counts_kept_messages_and_only_kept_messages_are_sampled() -> None:
    frame = pool({"a": 100, "b": 100}, french=lambda _, index: index % 2 == 0)
    result = sampler(max_messages=20, window_size=5, strata=1).sample(frame)
    assert 16 <= result.sampled.height <= 20
    assert set(result.sampled["language"]) == {"en"}
    assert result.dropped_in_drawn_windows[DropReason.OTHER_LANGUAGE] > 0


def test_language_identification_only_sees_messages_of_the_drawn_windows() -> None:
    detector = SpyDetector()
    frame = pool({f"c{k}": 100 for k in range(10)})
    result = sampler(max_messages=20, window_size=5, strata=1, detector=detector).sample(frame)
    assert result.sampled.height == 20
    assert result.examined_in_drawn_windows == len(detector.seen)
    assert result.examined_in_drawn_windows <= 40
    assert sum(result.dropped_in_drawn_windows.values()) == 0


def test_the_output_is_ordered_by_channel_then_time() -> None:
    shuffled = pool(CHANNELS).collect().sample(fraction=1.0, shuffle=True, seed=3)
    result = sampler(max_messages=60, window_size=4).sample(shuffled.lazy())
    ordered = result.sampled.sort("channel_id", "timestamp", "message_id")
    assert ids(result.sampled) == ids(ordered)


def test_the_same_seed_gives_the_same_sample_and_another_seed_a_different_one() -> None:
    frame = pool(CHANNELS)
    first = ids(sampler(max_messages=80, seed=1).sample(frame).sampled)
    again = ids(sampler(max_messages=80, seed=1).sample(frame).sampled)
    other = ids(sampler(max_messages=80, seed=2).sample(frame).sampled)
    assert first == again
    assert first != other


def test_the_input_row_order_does_not_change_the_sample() -> None:
    frame = pool(CHANNELS)
    shuffled = frame.collect().sample(fraction=1.0, shuffle=True, seed=9).lazy()
    first = ids(sampler(max_messages=80).sample(frame).sampled)
    second = ids(sampler(max_messages=80).sample(shuffled).sampled)
    assert first == second


def test_an_empty_pool_gives_an_empty_sample() -> None:
    result = sampler().sample(pool({}))
    assert result.sampled.height == 0
    assert "stratum" in result.sampled.columns
    assert result.eligible_per_stratum == {}
    assert result.examined_in_drawn_windows == 0


def test_more_strata_than_channels_leaves_strata_empty() -> None:
    result = sampler(max_messages=100, strata=10).sample(pool({"a": 10, "b": 20}))
    assert set(result.sampled["stratum"]) == {0, 5}
    assert result.sampled.height == 30


def test_a_frame_without_the_needed_columns_is_rejected() -> None:
    with pytest.raises(SamplingError, match="channel_id"):
        sampler().sample(pool(CHANNELS).drop("channel_id"))


@given(
    sizes=st.lists(st.integers(1, 40), min_size=1, max_size=12),
    cap=st.integers(1, 300),
    window=st.integers(1, 8),
    strata=st.integers(1, 6),
    seed=st.integers(0, 50),
)
@settings(deadline=None, max_examples=60)
def test_any_pool_gives_a_capped_whole_window_sample(
    sizes: list[int], cap: int, window: int, strata: int, seed: int
) -> None:
    channels = {f"c{k:02d}": size for k, size in enumerate(sizes)}
    cap = max(cap, window * strata)
    result = sampler(max_messages=cap, window_size=window, strata=strata, seed=seed).sample(
        pool(channels)
    )
    assert result.sampled.height <= cap
    per_stratum = dict(result.sampled.group_by("stratum").len().iter_rows())
    assert per_stratum == {s: n for s, n in result.sampled_per_stratum.items() if n}
    for channel, group in result.sampled.group_by("channel_id"):
        size = channels[channel[0]]
        positions = sorted(int(i.split("-")[1]) for i in group["message_id"])
        for window_index in {p // window for p in positions}:
            expected = min(window, size - window_index * window)
            assert sum(p // window == window_index for p in positions) == expected


@given(
    sizes=st.lists(st.integers(5, 40), min_size=1, max_size=6),
    cap=st.integers(1, 120),
    seed=st.integers(0, 20),
)
@settings(deadline=None, max_examples=40)
def test_with_language_drops_the_sample_is_capped_english_and_within_candidates(
    sizes: list[int], cap: int, seed: int
) -> None:
    channels = {f"c{k:02d}": size for k, size in enumerate(sizes)}
    frame = pool(channels, french=lambda _, index: index % 3 == 0)
    cap = max(cap, 8)
    result = sampler(max_messages=cap, window_size=4, strata=2, seed=seed).sample(frame)
    assert result.sampled.height <= cap
    assert set(result.sampled["language"]) <= {"en"}
    assert set(ids(result.sampled)) <= set(ids(frame.collect()))
    assert np.sum(list(result.eligible_per_stratum.values())) == sum(sizes)


def test_a_stratum_that_stops_on_an_overflowing_window_logs_a_warning_with_counts_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pool({"a": 100, "b": 100}, french=lambda _, index: index % 5 < 2)
    with caplog.at_level(logging.WARNING):
        result = sampler(max_messages=20, window_size=5, strata=1).sample(frame)
    warnings = [record.getMessage() for record in caplog.records]
    assert warnings == [
        "stratum 0 stopped on a window that would exceed its quota: "
        f"{result.sampled.height} of 20 messages"
    ]
    assert FRENCH not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        sampler(max_messages=20, window_size=5, strata=1).sample(pool({"a": 100}))
    assert not caplog.records
