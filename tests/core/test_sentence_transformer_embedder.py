from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from banhammer.core import sentence_transformer_embedder
from banhammer.core.embedding import Embedder, EmbedderSettings
from banhammer.core.errors import EmbeddingError
from banhammer.core.sentence_transformer_embedder import SentenceTransformerEmbedder
from embedder_double import FakeEmbedder

REVISION = "ab" * 20
SETTINGS = EmbedderSettings(model="org/model", revision=REVISION, batch_size=4, max_seq_length=64)
CPU = torch.device("cpu")


class FakeSentenceTransformer:
    def __init__(self, model_name_or_path: str, dimension: int | None, **kwargs: Any) -> None:
        self.model_name_or_path = model_name_or_path
        self.dimension = dimension
        self.kwargs = kwargs
        self.max_seq_length = 0
        self.encode_kwargs: dict[str, Any] = {}

    def get_embedding_dimension(self) -> int | None:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: Any) -> NDArray[np.float64]:
        self.encode_kwargs = kwargs
        vectors = np.tile(np.array([3.0, 4.0, 0.0]), (len(texts), 1))
        if kwargs.get("normalize_embeddings"):
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


class ModelLoader:
    def __init__(self) -> None:
        self.created: list[FakeSentenceTransformer] = []
        self.dimension: int | None = 3

    def __call__(self, model_name_or_path: str, **kwargs: Any) -> FakeSentenceTransformer:
        model = FakeSentenceTransformer(model_name_or_path, self.dimension, **kwargs)
        self.created.append(model)
        return model


@pytest.fixture(autouse=True)
def loader(monkeypatch: pytest.MonkeyPatch) -> ModelLoader:
    model_loader = ModelLoader()
    monkeypatch.setattr(sentence_transformer_embedder, "SentenceTransformer", model_loader)
    return model_loader


def test_model_is_loaded_offline_from_the_snapshot_without_remote_code(
    loader: ModelLoader, tmp_path: Path
) -> None:
    SentenceTransformerEmbedder(SETTINGS, torch.device("cpu"), tmp_path)
    (loaded,) = loader.created
    assert loaded.model_name_or_path == str(tmp_path)
    assert loaded.kwargs["trust_remote_code"] is False
    assert loaded.kwargs["local_files_only"] is True
    assert loaded.kwargs["device"] == "cpu"
    assert loaded.kwargs["model_kwargs"] == {"use_safetensors": True}
    assert "revision" not in loaded.kwargs


def test_the_configured_max_seq_length_is_applied(loader: ModelLoader, tmp_path: Path) -> None:
    SentenceTransformerEmbedder(SETTINGS, CPU, tmp_path)
    assert loader.created[0].max_seq_length == 64


def test_from_hub_downloads_the_pinned_snapshot_then_loads_it(
    monkeypatch: pytest.MonkeyPatch, loader: ModelLoader, tmp_path: Path
) -> None:
    requested: list[tuple[str, str]] = []

    def fake_download(model: str, revision: str) -> Path:
        requested.append((model, revision))
        return tmp_path

    monkeypatch.setattr(sentence_transformer_embedder, "download_snapshot", fake_download)
    SentenceTransformerEmbedder.from_hub(SETTINGS, CPU)
    assert requested == [("org/model", REVISION)]
    assert loader.created[0].model_name_or_path == str(tmp_path)


def test_encode_returns_l2_normalized_float32_rows(tmp_path: Path) -> None:
    result = SentenceTransformerEmbedder(SETTINGS, CPU, tmp_path).encode(["a", "b"])
    assert result.dtype == np.float32
    assert result.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(result[0], [0.6, 0.8, 0.0], atol=1e-6)


def test_encode_uses_the_configured_batch_size_and_asks_for_normalized_arrays(
    loader: ModelLoader, tmp_path: Path
) -> None:
    SentenceTransformerEmbedder(SETTINGS, CPU, tmp_path).encode(["a"])
    encode_kwargs = loader.created[0].encode_kwargs
    assert encode_kwargs["batch_size"] == 4
    assert encode_kwargs["normalize_embeddings"] is True
    assert encode_kwargs["convert_to_numpy"] is True
    assert encode_kwargs["show_progress_bar"] is False


def test_encoding_nothing_gives_an_empty_matrix_of_the_model_dimension(tmp_path: Path) -> None:
    result = SentenceTransformerEmbedder(SETTINGS, CPU, tmp_path).encode([])
    assert result.shape == (0, 3)
    assert result.dtype == np.float32


def test_encoding_nothing_fails_when_the_model_reports_no_dimension(
    loader: ModelLoader, tmp_path: Path
) -> None:
    loader.dimension = None
    embedder = SentenceTransformerEmbedder(SETTINGS, CPU, tmp_path)
    with pytest.raises(EmbeddingError, match="org/model"):
        embedder.encode([])


def test_the_double_and_the_real_embedder_satisfy_the_protocol(tmp_path: Path) -> None:
    embedders: Sequence[Embedder] = [
        FakeEmbedder({"a": [1.0, 0.0]}),
        SentenceTransformerEmbedder(SETTINGS, CPU, tmp_path),
    ]
    assert all(embedder.encode(["a"]).dtype == np.float32 for embedder in embedders)


def test_the_double_normalizes_and_records_its_calls() -> None:
    double = FakeEmbedder({"a": [3.0, 4.0], "b": [0.0, 2.0]})
    result = double.encode(["b", "a"])
    np.testing.assert_allclose(result, [[0.0, 1.0], [0.6, 0.8]], atol=1e-6)
    assert double.calls == [("b", "a")]
    assert double.encode([]).shape == (0, 2)
