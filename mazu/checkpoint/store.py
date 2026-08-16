import json
import os
import shutil
import tempfile
from pathlib import Path

from mazu.checkpoint.lock import ReentrantFileLock


class CheckpointIndex:
    """Flat, ordered, human-readable JSON list of checkpoint metadata.
    A SQLite table would be overkill at MVP scale (tens of checkpoints per session).

    Every mutating method takes the same cross-process file lock (see
    mazu/checkpoint/lock.py), so two `mazu` processes touching one project's
    checkpoints concurrently serialize instead of racing. `locked()` is exposed for
    CheckpointManager to wrap a larger read-modify-write sequence (e.g. computing
    the next checkpoint id from a load() and only appending it many steps later)
    in one atomic critical section, rather than just protecting each individual
    load()/save() call in isolation -- protecting them individually would still
    leave the "two processes both read the same next id" race open.
    """

    def __init__(self, checkpoints_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = checkpoints_dir / "index.json"
        self._lock = ReentrantFileLock(checkpoints_dir / ".index.lock")
        # Real bug caught live (via a flaky multi-process CI/local repro, not just
        # reasoning about it): the existence check here used to happen OUTSIDE the
        # lock ("if not self.index_path.exists(): self.save([])"). When two
        # processes construct a CheckpointIndex for the same fresh project at
        # nearly the same time, both can see the file doesn't exist yet -- but if
        # one process's save([]) call gets delayed by lock contention long enough
        # that the OTHER process has, in the meantime, fully run several real
        # snapshot() calls, the delayed save([]) still unconditionally writes an
        # EMPTY list once it finally gets the lock, silently wiping out everything
        # the other process already wrote. Re-checking existence AFTER acquiring
        # the lock (not before) closes that window: `locked()` is reentrant, so
        # this nests safely into save()'s own internal locking.
        with self.locked():
            if not self.index_path.exists():
                self.save([])

    def locked(self):
        """Context manager: `with self.index.locked(): ...`. Reentrant within the
        same process/thread, so nesting (e.g. snapshot() calling prune() while
        already holding it) is safe rather than deadlocking.
        """
        return self._lock.acquire()

    def load(self) -> list[dict]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def save(self, entries: list[dict]) -> None:
        """Writes via a temp file + os.replace so a concurrent load() (or a crash
        mid-write) can never observe a truncated/partial index.json -- the same
        atomic-write technique already used for write_file/edit_file in
        mazu/tools/fs.py, applied here for the same reason: without it, a process
        killed mid-write could corrupt index.json badly enough that every future
        `mazu timeline`/`checkpoint`/`rollback` call raises a JSONDecodeError until
        a human manually repairs the file.
        """
        with self.locked():
            fd, tmp_path = tempfile.mkstemp(
                dir=self.index_path.parent, prefix=".index.", suffix=".json.tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps(entries, indent=2))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.index_path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def append(self, entry: dict) -> None:
        with self.locked():
            entries = self.load()
            entries.append(entry)
            self.save(entries)

    def get(self, checkpoint_id: str) -> dict | None:
        for entry in self.load():
            if entry["id"] == checkpoint_id:
                return entry
        return None

    def last(self) -> dict | None:
        entries = self.load()
        return entries[-1] if entries else None

    def last_for_branch(self, branch: str) -> dict | None:
        """Like last(), but scoped to one git branch -- once checkpoints from
        multiple branches can coexist in this flat list, "most recently appended
        overall" (last()) is no longer the same thing as "most recent on the branch
        I'm currently on". Entries recorded before branch-awareness existed have no
        "branch" key at all; they're treated as belonging to every branch (the
        common case: they were all made on the one branch that existed back then,
        so they're still valid rollback targets on whatever branch that turned out
        to be).
        """
        matches = [e for e in self.load() if e.get("branch") in (branch, None)]
        return matches[-1] if matches else None

    def truncate_after(self, checkpoint_id: str) -> None:
        """Drops index entries after `checkpoint_id` AND deletes their on-disk folders.
        Without this, a rolled-back checkpoint id (e.g. cp_0003) can get reused by a
        later checkpoint while its old folder still exists on disk, causing
        shutil.copytree to collide with stale contents from the invalidated checkpoint.
        """
        with self.locked():
            entries = self.load()
            idx = next((i for i, e in enumerate(entries) if e["id"] == checkpoint_id), None)
            if idx is None:
                return
            removed = entries[idx + 1 :]
            self.save(entries[: idx + 1])
            for entry in removed:
                stale_dir = self.checkpoints_dir / entry["id"]
                if stale_dir.exists():
                    shutil.rmtree(stale_dir, ignore_errors=True)
