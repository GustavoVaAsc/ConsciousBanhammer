"""Download of the source dataset at its pinned commit.

Functional module: one stateless wrapper around the hub download.
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import filter_repo_objects

from banhammer.messages.constants import HUB_CACHE_DIRECTORY
from banhammer.messages.errors import SourceDownloadError
from banhammer.messages.settings import SourceSettings


def download_source(source: SourceSettings, target: Path) -> list[Path]:
    """Fetch the files of `source` matching its patterns at its pinned commit.

    The files go into `target / <revision>`, so files of another revision never mix in. The
    commit hash is validated by `SourceSettings`, so a branch or a tag cannot reach the hub.
    `etag_timeout` bounds the metadata request; the file transfer reads are bounded by the
    hub's own `HF_HUB_DOWNLOAD_TIMEOUT`, which `huggingface_hub` reads at import and which
    defaults to 10 seconds (`snapshot_download` has no parameter for it).

    Returns:
        The downloaded files matching `source.allow_patterns`, sorted.

    Raises:
        SourceDownloadError: The repository or revision does not exist, the download fails, no
            file matches the patterns, the directory holds a file the patterns do not match, or
            it holds a symlink (also a dangling one), which is never followed. The message names
            the exception class, never the hub's text.
    """
    directory = target / source.revision
    _reject_links(source, directory)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=source.repo_id,
                repo_type="dataset",
                revision=source.revision,
                allow_patterns=list(source.allow_patterns),
                local_dir=directory,
                etag_timeout=source.etag_timeout,
            )
        )
    except (OSError, HFValidationError) as error:
        message = (
            f"cannot download {source.repo_id} at revision {source.revision}: "
            f"{type(error).__name__}"
        )
        raise SourceDownloadError(message) from error
    _reject_links(source, directory)
    present = [
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and HUB_CACHE_DIRECTORY not in path.relative_to(snapshot).parts
    ]
    matched = set(filter_repo_objects(present, allow_patterns=list(source.allow_patterns)))
    if not matched:
        message = (
            f"no file of {source.repo_id} at {source.revision} matches {source.allow_patterns}"
        )
        raise SourceDownloadError(message)
    if extra := sorted(set(present) - matched):
        message = f"{source.repo_id} at {source.revision}: unexpected files {extra}"
        raise SourceDownloadError(message)
    return sorted(snapshot / name for name in matched)


def _reject_links(source: SourceSettings, directory: Path) -> None:
    if not directory.is_symlink() and not directory.exists():
        return
    entries = [directory, *directory.rglob("*")]
    links = [path.relative_to(directory.parent).as_posix() for path in entries if path.is_symlink()]
    if links:
        message = f"{source.repo_id} at {source.revision}: symlinks are not allowed: {links}"
        raise SourceDownloadError(message)
