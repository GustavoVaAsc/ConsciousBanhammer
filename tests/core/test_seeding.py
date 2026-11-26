import random
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from banhammer.core.seeding import Seeder


def draw_global_state() -> tuple[int, float, float]:
    return random.getrandbits(32), float(np.random.random()), float(torch.rand(1))


def test_seed_everything_makes_global_draws_repeat() -> None:
    seeder = Seeder(123)
    seeder.seed_everything()
    first = draw_global_state()
    seeder.seed_everything()
    assert draw_global_state() == first


def test_different_seeds_give_different_draws() -> None:
    Seeder(1).seed_everything()
    first = draw_global_state()
    Seeder(2).seed_everything()
    assert draw_global_state() != first


def test_seed_larger_than_numpy_accepts_is_reduced() -> None:
    Seeder(2**40 + 5).seed_everything()


Draws = Callable[[int, str], list[int]]


def numpy_draws(seed: int, stream: str) -> list[int]:
    return Seeder(seed).numpy_generator(stream).integers(2**32, size=4).tolist()


def torch_draws(seed: int, stream: str) -> list[int]:
    generator = Seeder(seed).torch_generator(stream)
    return torch.randint(2**31, (4,), generator=generator).tolist()


@pytest.mark.parametrize("draws", [numpy_draws, torch_draws])
def test_a_stream_is_reproducible_for_the_same_seed_and_name(
    draws: Draws,
) -> None:
    assert draws(5, "split") == draws(5, "split")


@pytest.mark.parametrize("draws", [numpy_draws, torch_draws])
def test_streams_with_different_names_draw_different_numbers(
    draws: Draws,
) -> None:
    assert draws(5, "split") != draws(5, "bootstrap")


@pytest.mark.parametrize("draws", [numpy_draws, torch_draws])
def test_the_same_stream_differs_across_seeds(draws: Draws) -> None:
    assert draws(5, "split") != draws(6, "split")


def test_a_stream_ignores_the_global_state_and_does_not_advance_it() -> None:
    expected = numpy_draws(5, "split")
    random.seed(0)
    np.random.seed(0)
    before = np.random.get_state()[1].tolist()
    assert numpy_draws(5, "split") == expected
    assert np.random.get_state()[1].tolist() == before


def test_seed_at_the_upper_bound_gives_a_stream() -> None:
    assert len(torch_draws(2**63 - 1, "split")) == 4


class NoiseDataset(Dataset[tuple[int, float]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> tuple[int, float]:
        return random.getrandbits(32), float(np.random.random())


def load_noise(seeder: Seeder) -> list[tuple[int, float]]:
    loader = DataLoader(
        NoiseDataset(),
        batch_size=2,
        num_workers=2,
        worker_init_fn=seeder.worker_init_fn,
        generator=seeder.torch_generator("loader"),
    )
    return [
        (int(python_draw), float(numpy_draw))
        for python_draws, numpy_draws in loader
        for python_draw, numpy_draw in zip(python_draws, numpy_draws, strict=True)
    ]


def test_dataloader_workers_repeat_across_runs_and_draw_distinct_streams() -> None:
    first = load_noise(Seeder(11))
    assert load_noise(Seeder(11)) == first
    python_draws, numpy_draws = zip(*first, strict=True)
    assert len(set(python_draws)) == len(first)
    assert len(set(numpy_draws)) == len(first)


@pytest.mark.gpu
def test_cuda_is_available_and_seeded_cuda_draws_repeat() -> None:
    assert torch.cuda.is_available()
    seeder = Seeder(3)
    seeder.seed_everything()
    first = torch.rand(4, device="cuda")
    seeder.seed_everything()
    second = torch.rand(4, device="cuda")
    assert torch.equal(first, second)
    assert float((first + 1).sum()) > 0
