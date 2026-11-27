"""Atomic replacement of a file, so a reader never sees a half-written artifact.

Functional module: one stateless context manager around `Path.replace`.
"""

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

NEW_FILE_MODE = 0o666


@contextmanager
def atomic_destination(path: Path) -> Iterator[Path]:
    """Yield a temporary path next to `path`; flush it and move it onto `path` if the block ends.

    The temporary file is in the same directory, so the final `Path.replace` is atomic on POSIX
    file systems, and it is fsynced first, so a crash cannot leave the new name pointing at
    unwritten data. The temporary file is created with mode `0o666`, so the kernel applies the
    process umask exactly as for any new file: the artifact is readable by the users a plain
    `open` would let read it (for instance services running under another uid), and the umask
    is never read or changed, which would race with other threads. If the block raises, `path`
    is left as it was and the temporary file is removed. A hard kill (SIGKILL, power loss) can
    leave a `.tmp` file behind; DVC clears a stage's outputs before it runs. The directory of
    `path` must exist.
    """
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    os.close(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, NEW_FILE_MODE))
    try:
        yield temporary
        with temporary.open("rb") as written:
            os.fsync(written.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
