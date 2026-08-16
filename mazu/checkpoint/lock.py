"""Cross-process file lock for `.mazu/checkpoints/index.json`, backed by the
`filelock` package.

Two `mazu` processes touching the same project's checkpoints concurrently (e.g. two
terminals running `mazu chat`/`mazu run` in the same directory) previously had no
coordination at all: a classic load-modify-save race on index.json where the second
process's save() could silently overwrite the first's, and -- worse -- both could
independently compute the same "next checkpoint id" from a stale read, producing two
checkpoints that collide on disk.

Implementation note: this module went through two hand-rolled attempts first --
OS-level byte-range locks (fcntl.flock/msvcrt.locking), then a directory-creation
scheme with a PID-liveness staleness check. Both looked correct in code review and
passed most runs, but a live multi-process CI run (and extensive local reproduction
afterward, including direct acquire/release tracing) caught both silently failing to
provide real mutual exclusion under contention on Windows -- two concurrent
processes ended up computing the identical "next checkpoint id" despite each
believing it held an exclusive lock. Rather than keep debugging a hand-rolled
primitive in this exact problem space, this now uses `filelock`, the small,
extremely widely-used library pip/virtualenv/tox rely on for the same purpose --
proven correct rather than re-proven from scratch.
"""

from pathlib import Path

from filelock import FileLock, Timeout

DEFAULT_LOCK_TIMEOUT = 30.0


class LockTimeoutError(TimeoutError):
    pass


class ReentrantFileLock:
    """Thin wrapper around filelock.FileLock exposing the same `.acquire()` ->
    context manager interface the rest of mazu/checkpoint/ already uses.
    FileLock is itself reentrant within the same thread (an internal counter, not
    a second OS-level acquisition), which is exactly what CheckpointManager.
    snapshot() needs: it holds the lock across git commands, id-assignment, file
    writes, and its own call to prune() (which independently takes the same lock
    when invoked standalone, e.g. `mazu checkpoint prune`) -- without reentrancy,
    that nested acquisition would deadlock a process against itself.
    """

    def __init__(self, lock_path: Path, timeout: float = DEFAULT_LOCK_TIMEOUT):
        # filelock manages its own lock file at this exact path (it does not need
        # to be pre-created, and unlike our old schemes, there's no separate
        # pid-file/directory bookkeeping for us to get wrong).
        self._lock = FileLock(str(lock_path) + ".filelock", timeout=timeout)

    def acquire(self):
        return _AcquireContext(self._lock)


class _AcquireContext:
    """Translates filelock's Timeout exception into our own LockTimeoutError, so
    callers (and tests) depend on a mazu-owned exception type, not filelock's.
    """

    def __init__(self, lock: FileLock):
        self._lock = lock

    def __enter__(self):
        try:
            self._lock.acquire()
        except Timeout as e:
            raise LockTimeoutError(
                f"Timed out waiting for the checkpoint index lock at "
                f"{self._lock.lock_file}. Another mazu process may be stuck; if "
                "none is actually running, this file is safe to delete."
            ) from e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False
