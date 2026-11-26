"""Commit-msg hook: a commit that touches `src/` must also change `CHANGELOG.md`.

Commits typed refactor, test, style or chore, which do not change behaviour, and merge commits are
exempt (AGENTS.md, definition of done, item 3).

The hook sees only the staged diff. On `git commit --amend` that is the newly staged files, not
the whole amended commit, so an amend that adds `src/` files must stage the `CHANGELOG.md` entry
again.

Functional module: entry point of a pre-commit hook and the pure helpers it composes.
"""

import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

EXEMPT_TYPES = frozenset({"refactor", "test", "style", "chore"})
SOURCE_PREFIX = "src/"
CHANGELOG_PATH = "CHANGELOG.md"
MERGE_PREFIX = "Merge "
COMMENT_PREFIX = "#"
SUBJECT_TYPE = re.compile(r"^(?P<type>[a-z]+)(\([^)]*\))?!?: ")
GIT_TIMEOUT_SECONDS = 30
REJECTION = (
    f"commit touches {SOURCE_PREFIX} but {CHANGELOG_PATH} has no staged change. "
    "Add an entry under [Unreleased] or, if the commit changes no behaviour, type it as one of: "
    + ", ".join(sorted(EXEMPT_TYPES))
    + ". On an amend, stage the changelog entry again.\n"
)


def subject_of(message: str) -> str:
    """Return the first line of `message` that is neither blank nor a git comment."""
    lines = (line.strip() for line in message.splitlines())
    return next((line for line in lines if line and not line.startswith(COMMENT_PREFIX)), "")


def is_exempt(subject: str) -> bool:
    """Whether a commit with this subject line may skip the changelog."""
    if subject.startswith(MERGE_PREFIX):
        return True
    match = SUBJECT_TYPE.match(subject)
    return match is not None and match["type"] in EXEMPT_TYPES


def rejection(message: str, staged_files: Sequence[str]) -> str | None:
    """Return the reason to reject the commit, or None when it may proceed."""
    touches_source = any(path.startswith(SOURCE_PREFIX) for path in staged_files)
    if not touches_source or CHANGELOG_PATH in staged_files or is_exempt(subject_of(message)):
        return None
    return REJECTION


def staged_files() -> list[str]:
    """List the paths staged in the current repository, relative to its root.

    Renames are listed as a deletion and an addition, so both sides count.
    """
    git = shutil.which("git")
    if git is None:
        message = "git is not on PATH"
        raise FileNotFoundError(message)
    result = subprocess.run(  # noqa: S603
        [git, "diff", "--cached", "--name-only", "--no-renames", "-z"],
        capture_output=True,
        check=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    return [path for path in result.stdout.split("\0") if path]


def main(argv: Sequence[str]) -> int:
    """Hook entry point; `argv[1]` is the commit message file git passes to commit-msg hooks."""
    message = Path(argv[1]).read_text(encoding="utf-8")
    reason = rejection(message, staged_files())
    if reason is not None:
        sys.stderr.write(reason)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
