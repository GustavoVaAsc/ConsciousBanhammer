"""Root pytest configuration: registers the `gpu` and `integration` markers."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the project markers so `--strict-markers` accepts them."""
    config.addinivalue_line("markers", "gpu: needs a CUDA device; run on the GPU box only")
    config.addinivalue_line(
        "markers", "integration: calls real backends and may spend API budget; run only on request"
    )
