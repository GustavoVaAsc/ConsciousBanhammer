"""Download of a Hugging Face model at a pinned commit, restricted to non-pickle files.

Functional module: one stateless wrapper around the hub download.

Only safetensors, JSON and text files are fetched, so no pickle (`*.bin`, `*.pt`, `*.pkl`)
can reach the disk (H16). The caller loads the returned local directory offline.
"""

import re
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import filter_repo_objects

from banhammer.core.constants import COMMIT_HASH_PATTERN, MODEL_SNAPSHOT_PATTERNS
from banhammer.core.errors import ModelSnapshotError


def download_snapshot(model: str, revision: str) -> Path:
    """Fetch the safetensors, JSON and text files of `model` at commit `revision`.

    Returns:
        The local snapshot directory; files already in the Hugging Face cache are reused.

    Raises:
        ModelSnapshotError: `revision` is not a full commit hash, the model id is malformed, the
            repository or revision does not exist, the download fails, or the snapshot holds a
            file outside the allowed patterns (e.g. a stale pickle from an earlier download).
    """
    if not re.fullmatch(COMMIT_HASH_PATTERN, revision):
        message = f"revision of {model} must be a full lowercase commit hash, got {revision!r}"
        raise ModelSnapshotError(message)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model, revision=revision, allow_patterns=list(MODEL_SNAPSHOT_PATTERNS)
            )
        )
    except (OSError, HFValidationError) as error:
        message = f"cannot download {model} at revision {revision}: {error}"
        raise ModelSnapshotError(message) from error
    verify_snapshot(snapshot)
    return snapshot


def verify_snapshot(snapshot: Path) -> None:
    """Check that a local model directory holds only safetensors, JSON and text files.

    `download_snapshot` calls it on what it fetched. Code that loads a model from a directory
    it was handed calls it first, so a hand-made directory cannot skip the allow-list and
    smuggle in a pickle (H16). A symlink to a directory is rejected, because the walk does not
    follow it and the files behind it would go unchecked; the Hub cache links files only, so
    symlinks to files stay allowed (their names are checked like any other).

    Raises:
        ModelSnapshotError: `snapshot` is not a directory, holds a symlink to a directory, or
            holds a file outside the allowed patterns.
    """
    if not snapshot.is_dir():
        message = f"model snapshot {snapshot} is not a directory"
        raise ModelSnapshotError(message)
    entries = list(snapshot.rglob("*"))
    linked_directories = sorted(
        path.relative_to(snapshot).as_posix()
        for path in entries
        if path.is_symlink() and path.is_dir()
    )
    if linked_directories:
        message = (
            f"model snapshot {snapshot} holds symlinks to directories, whose files are not "
            f"checked: {linked_directories}"
        )
        raise ModelSnapshotError(message)
    present = [path.relative_to(snapshot).as_posix() for path in entries if not path.is_dir()]
    allowed = set(filter_repo_objects(present, allow_patterns=list(MODEL_SNAPSHOT_PATTERNS)))
    unexpected = sorted(set(present) - allowed)
    if unexpected:
        message = (
            f"model snapshot {snapshot} holds files outside the allowed patterns: {unexpected}"
        )
        raise ModelSnapshotError(message)
