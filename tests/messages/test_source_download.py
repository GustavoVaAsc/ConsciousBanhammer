from pathlib import Path

import pytest
from huggingface_hub.errors import HFValidationError

from banhammer.messages import SourceDownloadError, SourceSettings, download_source
from banhammer.messages.settings import SourceColumns

REVISION = "cd" * 20


def source(patterns: tuple[str, ...] = ("*.parquet",)) -> SourceSettings:
    columns = SourceColumns(
        message_id="id",
        channel_id="c",
        parent_message_id=None,
        timestamp="t",
        text="x",
        is_bot=None,
    )
    return SourceSettings(
        repo_id="org/data",
        revision=REVISION,
        allow_patterns=patterns,
        etag_timeout=7.5,
        columns=columns,
    )


def fake_hub(
    monkeypatch: pytest.MonkeyPatch, files: list[str], stale: tuple[str, ...] = ()
) -> dict[str, object]:
    """A hub that writes `files` (as it would for matching patterns) and finds `stale` on disk."""
    seen: dict[str, object] = {}

    def snapshot_download(**kwargs: object) -> str:
        seen.update(kwargs)
        root = Path(str(kwargs["local_dir"]))
        for name in [*files, *stale, ".cache/huggingface/download/a.parquet.metadata"]:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(b"")
        return str(root)

    monkeypatch.setattr("banhammer.messages.source_download.snapshot_download", snapshot_download)
    return seen


def test_the_pinned_commit_patterns_and_timeout_are_passed_to_the_hub_in_a_revision_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = fake_hub(monkeypatch, ["a.parquet", "data/b.parquet"])
    files = download_source(source(), tmp_path)
    assert seen["repo_id"] == "org/data"
    assert seen["repo_type"] == "dataset"
    assert seen["revision"] == REVISION
    assert seen["allow_patterns"] == ["*.parquet"]
    assert seen["etag_timeout"] == 7.5
    assert seen["local_dir"] == tmp_path / REVISION
    assert [path.relative_to(tmp_path / REVISION).as_posix() for path in files] == [
        "a.parquet",
        "data/b.parquet",
    ]


def test_patterns_can_reach_into_subdirectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_hub(monkeypatch, ["data/b.parquet", "data/c.parquet"])
    files = download_source(source(("data/*.parquet",)), tmp_path)
    assert [path.name for path in files] == ["b.parquet", "c.parquet"]


def test_no_matching_file_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_hub(monkeypatch, [])
    with pytest.raises(SourceDownloadError, match="matches"):
        download_source(source(), tmp_path)


def test_a_file_the_patterns_do_not_match_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_hub(monkeypatch, ["a.parquet"], stale=("old.pkl", "notes/README.md"))
    with pytest.raises(SourceDownloadError, match=r"unexpected files.*old\.pkl"):
        download_source(source(), tmp_path)


@pytest.mark.parametrize("error", [OSError("hub text /secret/path"), HFValidationError("bad id")])
def test_hub_failures_name_the_class_repo_and_revision_but_not_the_hub_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def failing(**_: object) -> str:
        raise error

    monkeypatch.setattr("banhammer.messages.source_download.snapshot_download", failing)
    with pytest.raises(SourceDownloadError, match=REVISION) as raised:
        download_source(source(), tmp_path)
    assert type(error).__name__ in str(raised.value)
    assert "org/data" in str(raised.value)
    assert str(error) not in str(raised.value)


def test_symlinks_in_the_revision_directory_are_rejected_before_and_after_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = fake_hub(monkeypatch, ["a.parquet"])
    directory = tmp_path / REVISION
    directory.mkdir()
    (directory / "dangling.parquet").symlink_to(tmp_path / "missing.parquet")
    with pytest.raises(SourceDownloadError, match=r"symlinks.*dangling\.parquet"):
        download_source(source(), tmp_path)
    assert not seen


def test_a_symlink_written_by_the_hub_step_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"")

    def snapshot_download(**kwargs: object) -> str:
        root = Path(str(kwargs["local_dir"]))
        root.mkdir(parents=True)
        (root / "a.parquet").symlink_to(outside)
        return str(root)

    monkeypatch.setattr("banhammer.messages.source_download.snapshot_download", snapshot_download)
    with pytest.raises(SourceDownloadError, match="symlinks"):
        download_source(source(), tmp_path / "raw")


def test_a_symlinked_revision_directory_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / REVISION).symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(SourceDownloadError, match="symlinks"):
        download_source(source(), tmp_path)
