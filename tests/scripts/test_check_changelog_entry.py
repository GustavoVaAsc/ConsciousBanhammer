import shutil
import subprocess
from pathlib import Path

import pytest

import check_changelog_entry as gate

SOURCE = "src/banhammer/core/seeding.py"
CHANGELOG = "CHANGELOG.md"


@pytest.mark.parametrize("commit_type", ["refactor", "test", "style", "chore"])
def test_exempt_types_may_touch_src_without_a_changelog_entry(commit_type: str) -> None:
    assert gate.rejection(f"{commit_type}: tidy\n", [SOURCE]) is None


@pytest.mark.parametrize("subject", ["refactor(core): tidy", "chore!: drop", "test(core)!: x"])
def test_scope_and_breaking_marker_do_not_hide_the_type(subject: str) -> None:
    assert gate.rejection(subject + "\n", [SOURCE]) is None


@pytest.mark.parametrize("commit_type", ["feat", "fix", "docs", "build(deps)", "perf", "ci"])
def test_other_types_touching_src_without_a_changelog_entry_are_rejected(
    commit_type: str,
) -> None:
    assert gate.rejection(f"{commit_type}: change\n", [SOURCE]) is not None


def test_changelog_staged_with_src_passes() -> None:
    assert gate.rejection("feat: add seeding\n", [SOURCE, CHANGELOG]) is None


def test_commit_not_touching_src_passes() -> None:
    assert gate.rejection("feat: docs only\n", ["docs/dependencies.md", "tests/a.py"]) is None


def test_path_merely_containing_src_is_not_source() -> None:
    assert gate.rejection("feat: x\n", ["docs/src/notes.md", "tests/src/a.py"]) is None


def test_commit_with_no_staged_files_passes() -> None:
    assert gate.rejection("feat: x\n", []) is None


def test_message_without_a_conventional_type_is_not_exempt() -> None:
    assert gate.rejection("update seeding\n", [SOURCE]) is not None


def test_a_type_only_in_the_body_does_not_exempt_the_commit() -> None:
    assert gate.rejection("feat: x\n\nrefactor: inside\n", [SOURCE]) is not None


def test_merge_commits_are_exempt() -> None:
    assert gate.rejection("Merge pull request #2 from a/b\n", [SOURCE]) is None


def test_git_comment_lines_and_blank_lines_before_the_subject_are_skipped() -> None:
    message = "\n# Please enter the commit message\nchore: tidy\n"
    assert gate.subject_of(message) == "chore: tidy"
    assert gate.rejection(message, [SOURCE]) is None


def test_empty_message_has_an_empty_subject() -> None:
    assert gate.subject_of("# only a comment\n") == ""


def git(repository: Path, *arguments: str) -> None:
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run([executable, *arguments], cwd=repository, check=True, capture_output=True)  # noqa: S603


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    git(tmp_path, "init", "--quiet")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def stage(repository: Path, relative: str) -> None:
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")
    git(repository, "add", relative)


def test_staged_files_lists_only_what_is_staged(repository: Path) -> None:
    stage(repository, SOURCE)
    (repository / "untracked.py").write_text("x\n", encoding="utf-8")
    assert gate.staged_files() == [SOURCE]


def test_main_rejects_a_src_commit_without_changelog(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stage(repository, SOURCE)
    message_file = repository / "COMMIT_EDITMSG"
    message_file.write_text("feat: add seeding\n", encoding="utf-8")
    assert gate.main(["check_changelog_entry.py", str(message_file)]) == 1
    assert CHANGELOG in capsys.readouterr().err


def test_main_accepts_a_src_commit_with_changelog(repository: Path) -> None:
    stage(repository, SOURCE)
    stage(repository, CHANGELOG)
    message_file = repository / "COMMIT_EDITMSG"
    message_file.write_text("feat: add seeding\n", encoding="utf-8")
    assert gate.main(["check_changelog_entry.py", str(message_file)]) == 0


def commit(repository: Path, message: str) -> None:
    git(repository, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", message)


def test_a_rename_out_of_src_lists_both_paths(repository: Path) -> None:
    stage(repository, SOURCE)
    commit(repository, "feat: add seeding")
    (repository / "docs").mkdir()
    git(repository, "mv", SOURCE, "docs/seeding.py")
    assert sorted(gate.staged_files()) == ["docs/seeding.py", SOURCE]
    assert gate.rejection("feat: move\n", gate.staged_files()) is not None


def test_amend_is_checked_against_the_newly_staged_files_only(repository: Path) -> None:
    stage(repository, "README")
    commit(repository, "chore: root")
    stage(repository, SOURCE)
    stage(repository, CHANGELOG)
    commit(repository, "feat: add seeding")
    stage(repository, "src/banhammer/core/errors.py")
    assert gate.staged_files() == ["src/banhammer/core/errors.py"]
    assert "stage the changelog entry again" in (
        gate.rejection("feat: x\n", gate.staged_files()) or ""
    )
