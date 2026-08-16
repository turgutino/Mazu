"""Cross-process, cross-platform advisory file lock for `.mazu/checkpoints/index.json`.

Two `mazu` processes touching the same project's checkpoints concurrently (e.g. two
terminals running `mazu chat`/`mazu run` in the same directory) previously had no
coordination at all: a classic load-modify-save race on index.json where the second
process's save() could silently overwrite the first's, and -- worse -- both could
independently compute the same "next checkpoint id" from a stale read, producing two
checkpoints that collide on disk.

Uses the OS's own advisory file lock (fcntl.flock on POSIX, msvcrt.locking on
Windows) rather than a lock-file-existence scheme (`open(path, 'x')` + delete): an
OS-level lock is automatically released by the kernel when the holding process exits
for ANY reason, including a crash or `kill -9` -- a lock-file-existence scheme would
instead leave a stale lock file behind forever in that case, permanently wedging
every future checkpoint operation until a human manually deletes it.
"""

import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_LOCK_TIMEOUT = 30.0
_POLL_INTERVAL = 0.05


class LockTimeoutError(TimeoutError):
    pass


if sys.platform == "win32":
    import msvcrt

    def _try_acquire(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_acquire(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _os_lock(lock_path: Path, timeout: float):
    """Blocks (polling) until the OS-level lock on `lock_path` is acquired, or
    raises LockTimeoutError after `timeout` seconds. msvcrt.locking requires the
    file to have at least one byte to lock, hence the 'a+b' open + guaranteed byte.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")
    try:
        if os.fstat(f.fileno()).st_size == 0:
            f.write(b"0")
            f.flush()
        fd = f.fileno()
        os.lseek(fd, 0, os.SEEK_SET)
        deadline = time.monotonic() + timeout
        while not _try_acquire(fd):
            if time.monotonic() >= deadline:
                raise LockTimeoutError(
                    f"Timed out after {timeout}s waiting for the checkpoint index "
                    f"lock at {lock_path}. Another mazu process may be stuck; if "
                    "none is actually running, this file is safe to delete."
                )
            time.sleep(_POLL_INTERVAL)
        try:
            yield
        finally:
            _release(fd)
    finally:
        f.close()


class ReentrantFileLock:
    """A `with` context manager combining the cross-process OS lock above with
    same-process, same-thread reentrancy tracking. Reentrancy matters because
    CheckpointManager.snapshot() needs to hold the lock across id-assignment,
    file writes, and its own call to prune() (which independently takes the same
    lock when invoked standalone, e.g. `mazu checkpoint prune`) -- without
    reentrancy, that nested acquisition would deadlock a process against itself,
    since flock/msvcrt.locking are not reentrant across separate acquisitions
    even from the same process.
    """

    def __init__(self, lock_path: Path, timeout: float = DEFAULT_LOCK_TIMEOUT):
        self.lock_path = lock_path
        self.timeout = timeout
        self._local = threading.local()

    @contextmanager
    def acquire(self):
        depth = getattr(self._local, "depth", 0)
        if depth > 0:
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth = depth
            return

        self._local.depth = 1
        try:
            with _os_lock(self.lock_path, self.timeout):
                yield
        finally:
            self._local.depth = 0
