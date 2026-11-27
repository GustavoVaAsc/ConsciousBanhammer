from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError
from huggingface_hub.utils import filter_repo_objects

from banhammer.core import model_snapshot
from banhammer.core.errors import BanhammerError, ModelSnapshotError
from banhammer.core.model_snapshot import download_snapshot, verify_snapshot

REVISION = "ab" * 20
PICKLE_FILES = ["model.bin", "2_Dense/pytorch_model.bin", "weights.pt", "state.pkl", "model.pth"]
WANTED_FILES = ["model.safetensors", "2_Dense/model.safetensors", "config.json", "vocab.txt"]


def snapshot_at(path: Path) -> Callable[..., str]:
    def fake_snapshot_download(**kwargs: Any) -> str:
        return str(path)

    return fake_snapshot_download


def allowed_by_the_hub(names: list[str], patterns: list[str]) -> list[str]:
    return list(filter_repo_objects(names, allow_patterns=patterns))


def test_snapshot_is_requested_at_the_pinned_revision_with_non_pickle_patterns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(model_snapshot, "snapshot_download", fake_snapshot_download)
    assert download_snapshot("org/model", REVISION) == tmp_path
    (call,) = calls
    assert call["repo_id"] == "org/model"
    assert call["revision"] == REVISION
    patterns = call["allow_patterns"]
    assert allowed_by_the_hub(WANTED_FILES, patterns) == WANTED_FILES
    assert allowed_by_the_hub(PICKLE_FILES, patterns) == []


@pytest.mark.parametrize(
    "error", [LocalEntryNotFoundError("offline"), HFValidationError("bad id"), OSError("disk")]
)
def test_download_failures_become_snapshot_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def failing(**kwargs: Any) -> str:
        raise error

    monkeypatch.setattr(model_snapshot, "snapshot_download", failing)
    with pytest.raises(ModelSnapshotError, match="org/model") as raised:
        download_snapshot("org/model", REVISION)
    assert raised.value.__cause__ is error
    assert isinstance(raised.value, BanhammerError)


@pytest.mark.parametrize("revision", ["main", "", "v1.0", "AB" * 20, "ab" * 19, "ab" * 20 + "\n"])
def test_a_revision_that_is_not_a_commit_hash_is_rejected_before_any_download(
    monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    def must_not_run(**kwargs: Any) -> str:
        raise AssertionError(kwargs)

    monkeypatch.setattr(model_snapshot, "snapshot_download", must_not_run)
    with pytest.raises(ModelSnapshotError, match="commit hash"):
        download_snapshot("org/model", revision)


@pytest.mark.parametrize(
    "stale", ["pytorch_model.bin", "2_Dense/pytorch_model.bin", "w.pt", "x.pkl"]
)
def test_a_snapshot_holding_a_stale_pickle_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stale: str
) -> None:
    for name in [*WANTED_FILES, stale]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(model_snapshot, "snapshot_download", snapshot_at(tmp_path))
    with pytest.raises(ModelSnapshotError, match=stale):
        download_snapshot("org/model", REVISION)


def test_a_snapshot_of_allowed_files_only_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in WANTED_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "1_Pooling").mkdir()
    monkeypatch.setattr(model_snapshot, "snapshot_download", snapshot_at(tmp_path))
    assert download_snapshot("org/model", REVISION) == tmp_path


@pytest.mark.parametrize("stale", PICKLE_FILES)
def test_verify_snapshot_rejects_any_file_outside_the_allowed_patterns(
    tmp_path: Path, stale: str
) -> None:
    for name in [*WANTED_FILES, stale]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(ModelSnapshotError, match=stale):
        verify_snapshot(tmp_path)


def test_verify_snapshot_accepts_allowed_files_and_an_empty_directory(tmp_path: Path) -> None:
    verify_snapshot(tmp_path)
    for name in WANTED_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_a_path_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ModelSnapshotError, match="not a directory"):
        verify_snapshot(tmp_path / "missing")


def test_verify_snapshot_rejects_a_symlinked_directory_that_could_hide_a_pickle(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "pytorch_model.bin").write_bytes(b"x")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"x")
    (snapshot / "hidden").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ModelSnapshotError, match=r"symlink.*hidden"):
        verify_snapshot(snapshot)


def test_verify_snapshot_allows_symlinked_files_as_in_the_hub_cache(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_bytes(b"x")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(blobs / "abc")
    verify_snapshot(snapshot)
