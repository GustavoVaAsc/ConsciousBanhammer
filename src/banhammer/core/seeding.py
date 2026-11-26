"""Seeding of every random number source a stage draws from (H12)."""

import hashlib
import random

import numpy as np
import torch

from banhammer.core.constants import NUMPY_SEED_MODULUS, STREAM_KEY_BYTES


class Seeder:
    """Seeds Python, NumPy, torch and CUDA from one integer and hands out named random streams."""

    def __init__(self, seed: int) -> None:
        """Remember the seed, taken from the `core` section of `params.yaml`."""
        self._seed = seed

    def seed_everything(self) -> None:
        """Seed the global generators of Python, NumPy, torch and every CUDA device.

        Call once at the start of each stage.
        """
        random.seed(self._seed)
        np.random.seed(self._seed % NUMPY_SEED_MODULUS)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)

    def numpy_generator(self, stream: str) -> np.random.Generator:
        """A NumPy generator for the named stream.

        The same seed and stream always give the same draws; different streams are independent,
        so a split and a bootstrap that use different names never share random numbers.
        """
        return np.random.default_rng(self._stream_sequence(stream))

    def torch_generator(self, stream: str) -> torch.Generator:
        """A CPU torch generator for the named stream, e.g. for a `DataLoader`.

        Streams behave as in `numpy_generator`.
        """
        (state,) = self._stream_sequence(stream).generate_state(1, dtype=np.uint64)
        return torch.Generator().manual_seed(int(state))

    # https://pytorch.org/docs/stable/data.html: DataLoader fixes the worker_init_fn signature
    def worker_init_fn(self, worker_id: int) -> None:  # noqa: ARG002
        """Seed Python and NumPy inside a `DataLoader` worker from torch's per-worker seed.

        torch derives the worker seed from the loader's generator and the worker id, so passing
        `torch_generator(...)` to the `DataLoader` makes the workers reproducible.
        """
        worker_seed = torch.initial_seed() % NUMPY_SEED_MODULUS
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def _stream_sequence(self, stream: str) -> np.random.SeedSequence:
        digest = hashlib.sha256(stream.encode("utf-8")).digest()
        key = int.from_bytes(digest[:STREAM_KEY_BYTES], "big")
        return np.random.SeedSequence(entropy=self._seed, spawn_key=(key,))
