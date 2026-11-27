from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from banhammer.core.errors import ModelSnapshotError
from banhammer.rules import cross_encoder_entailment_scorer
from banhammer.rules.cross_encoder_entailment_scorer import CrossEncoderEntailmentScorer
from banhammer.rules.entailment import EntailmentScorer
from banhammer.rules.errors import EntailmentError
from banhammer.rules.settings import NliSettings
from entailment_double import FakeEntailmentScorer

REVISION = "ab" * 20
SETTINGS = NliSettings(model="org/nli", revision=REVISION, batch_size=4, max_length=64)
CPU = torch.device("cpu")
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


class FakeCrossEncoder:
    def __init__(self, path: str, id2label: Mapping[int, str], **kwargs: Any) -> None:
        self.path = path
        self.kwargs = kwargs
        self.config = SimpleNamespace(id2label=dict(id2label))
        self.predict_calls: list[tuple[list[Any], dict[str, Any]]] = []

    def predict(self, pairs: Sequence[Any], **kwargs: Any) -> NDArray[np.float32]:
        self.predict_calls.append((list(pairs), kwargs))
        rows = [[0.7, 0.2, 0.1] if "same" in pair[1] else [0.1, 0.3, 0.6] for pair in pairs]
        return np.array(rows, dtype=np.float32)


class Loader:
    def __init__(self) -> None:
        self.id2label: Mapping[int, str] = LABELS
        self.created: list[FakeCrossEncoder] = []

    def __call__(self, path: str, **kwargs: Any) -> FakeCrossEncoder:
        model = FakeCrossEncoder(path, self.id2label, **kwargs)
        self.created.append(model)
        return model


@pytest.fixture(autouse=True)
def loader(monkeypatch: pytest.MonkeyPatch) -> Loader:
    model_loader = Loader()
    monkeypatch.setattr(cross_encoder_entailment_scorer, "CrossEncoder", model_loader)
    return model_loader


def scorer(tmp_path: Path) -> CrossEncoderEntailmentScorer:
    return CrossEncoderEntailmentScorer(SETTINGS, CPU, torch.float32, tmp_path)


def test_the_model_is_loaded_offline_from_safetensors_without_remote_code(
    loader: Loader, tmp_path: Path
) -> None:
    scorer(tmp_path)
    (model,) = loader.created
    assert model.path == str(tmp_path)
    assert model.kwargs["trust_remote_code"] is False
    assert model.kwargs["local_files_only"] is True
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["max_length"] == 64
    assert model.kwargs["model_kwargs"] == {"use_safetensors": True, "dtype": torch.float32}
    assert "revision" not in model.kwargs


def test_from_hub_downloads_the_pinned_snapshot_then_loads_it(
    monkeypatch: pytest.MonkeyPatch, loader: Loader, tmp_path: Path
) -> None:
    requested: list[tuple[str, str]] = []

    def fake_download(model: str, revision: str) -> Path:
        requested.append((model, revision))
        return tmp_path

    monkeypatch.setattr(cross_encoder_entailment_scorer, "download_snapshot", fake_download)
    CrossEncoderEntailmentScorer.from_hub(SETTINGS, CPU, torch.float32)
    assert requested == [("org/nli", REVISION)]
    assert loader.created[0].path == str(tmp_path)


def test_the_entailment_column_is_returned_as_float32_in_input_order(tmp_path: Path) -> None:
    result = scorer(tmp_path).entailment_probabilities(["p1", "p2"], ["same", "other"])
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [0.7, 0.1])


def test_pairs_are_passed_premise_first_with_the_configured_batch_size(
    loader: Loader, tmp_path: Path
) -> None:
    scorer(tmp_path).entailment_probabilities(["p1", "p2"], ["h1", "h2"])
    pairs, kwargs = loader.created[0].predict_calls[0]
    assert pairs == [("p1", "h1"), ("p2", "h2")]
    assert kwargs["batch_size"] == 4
    assert kwargs["apply_softmax"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["show_progress_bar"] is False


def test_the_label_is_found_by_name_whatever_its_index(loader: Loader, tmp_path: Path) -> None:
    loader.id2label = {0: "contradiction", 1: "Entailment", 2: "neutral"}
    result = scorer(tmp_path).entailment_probabilities(["p"], ["same"])
    np.testing.assert_allclose(result, [0.2])


@pytest.mark.parametrize(
    "labels",
    [{0: "LABEL_0", 1: "LABEL_1"}, {0: "entailment", 1: "entailment", 2: "neutral"}],
)
def test_a_model_without_exactly_one_entailment_label_is_rejected(
    loader: Loader, tmp_path: Path, labels: Mapping[int, str]
) -> None:
    loader.id2label = labels
    with pytest.raises(EntailmentError, match="org/nli"):
        scorer(tmp_path)


def test_no_pairs_give_an_empty_array_without_calling_the_model(
    loader: Loader, tmp_path: Path
) -> None:
    result = scorer(tmp_path).entailment_probabilities([], [])
    assert result.shape == (0,)
    assert result.dtype == np.float32
    assert loader.created[0].predict_calls == []


def test_premises_and_hypotheses_must_have_the_same_length(tmp_path: Path) -> None:
    with pytest.raises(EntailmentError, match=r"2.*1"):
        scorer(tmp_path).entailment_probabilities(["a", "b"], ["c"])


def test_the_double_and_the_real_scorer_satisfy_the_protocol(tmp_path: Path) -> None:
    scorers: Sequence[EntailmentScorer] = [FakeEntailmentScorer({}), scorer(tmp_path)]
    assert all(s.entailment_probabilities(["a"], ["b"]).dtype == np.float32 for s in scorers)


def test_the_double_looks_pairs_up_and_records_its_calls() -> None:
    double = FakeEntailmentScorer({("a", "b"): 0.9}, default=0.1)
    result = double.entailment_probabilities(["a", "b"], ["b", "a"])
    np.testing.assert_allclose(result, [0.9, 0.1], atol=1e-6)
    assert double.calls == [(("a", "b"), ("b", "a"))]


def test_a_directory_holding_a_pickle_is_rejected_before_the_model_is_loaded(
    loader: Loader, tmp_path: Path
) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(ModelSnapshotError, match=r"pytorch_model\.bin"):
        scorer(tmp_path)
    assert loader.created == []
