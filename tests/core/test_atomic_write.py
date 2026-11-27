import os
import stat
from pathlib import Path

import pytest

from banhammer.core.atomic_write import atomic_destination


def test_the_content_appears_at_the_target_only_when_the_block_ends(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    with atomic_destination(target) as temporary:
        assert temporary.parent == tmp_path
        assert temporary != target
        temporary.write_text("new", encoding="utf-8")
        assert not target.exists()
    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.iterdir()) == [target]


def test_an_existing_file_is_replaced(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    with atomic_destination(target) as temporary:
        temporary.write_text("new", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "new"


def test_a_failure_in_the_block_keeps_the_old_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"), atomic_destination(target) as temporary:
        temporary.write_text("partial", encoding="utf-8")
        message = "boom"
        raise RuntimeError(message)
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_a_failure_with_no_previous_file_leaves_nothing(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    with pytest.raises(RuntimeError, match="boom"), atomic_destination(target):
        message = "boom"
        raise RuntimeError(message)
    assert list(tmp_path.iterdir()) == []


def test_the_file_gets_the_mode_a_new_file_gets_under_the_process_umask(tmp_path: Path) -> None:
    previous = os.umask(0o027)
    try:
        target = tmp_path / "out.txt"
        with atomic_destination(target) as temporary:
            temporary.write_text("new", encoding="utf-8")
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
        plain = tmp_path / "plain.txt"
        plain.write_text("x", encoding="utf-8")
        assert stat.S_IMODE(plain.stat().st_mode) == stat.S_IMODE(target.stat().st_mode)
    finally:
        os.umask(previous)


def test_the_content_is_flushed_to_disk_before_it_is_moved_into_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    real_fsync, real_replace = os.fsync, Path.replace

    def recording_fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    def recording_replace(self: Path, target: Path) -> Path:
        events.append("replace")
        return real_replace(self, target)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(Path, "replace", recording_replace)
    with atomic_destination(tmp_path / "out.txt") as temporary:
        temporary.write_text("new", encoding="utf-8")
    assert events == ["fsync", "replace"]
