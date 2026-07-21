import json
import shutil
from pathlib import Path


class CheckpointIndex:
    """Flat, ordered, human-readable JSON list of checkpoint metadata.
    A SQLite table would be overkill at MVP scale (tens of checkpoints per session).
    """

    def __init__(self, checkpoints_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = checkpoints_dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text("[]", encoding="utf-8")

    def load(self) -> list[dict]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def save(self, entries: list[dict]) -> None:
        self.index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def append(self, entry: dict) -> None:
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
