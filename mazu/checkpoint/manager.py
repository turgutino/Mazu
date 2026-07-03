import json
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mazu.checkpoint.store import CheckpointIndex

DEFAULT_RETENTION = 50


def _git(root: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )


def _backup_sqlite(src_path: Path, dst_path: Path) -> None:
    """Uses SQLite's own online backup API instead of a raw file copy. A plain
    shutil.copy2 could copy a partially-written file if a write transaction is in
    flight on the source at the exact moment of the checkpoint; the backup API
    produces a consistent snapshot even under concurrent writes.
    """
    src_conn = sqlite3.connect(src_path)
    dst_conn = sqlite3.connect(dst_path)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


class CheckpointManager:
    """Each checkpoint = a git commit + a copy of memory.db + the skills directory +
    the conversation at that point. Rollback is a destructive, linear `git reset
    --hard` equivalent for all of these — no branching tree yet (see project roadmap
    for that future work). `retention` bounds how many checkpoints' worth of
    memory.db/skills/conversation.json copies are kept on disk at once — without
    this, `.mazu/checkpoints/` grows forever. Git history itself is never pruned,
    only our redundant snapshot copies.
    """

    def __init__(self, root: Path, retention: int = DEFAULT_RETENTION):
        self.root = root
        self.checkpoints_dir = root / ".mazu" / "checkpoints"
        self.index = CheckpointIndex(self.checkpoints_dir)
        self.memory_db_path = root / ".mazu" / "memory.db"
        self.skills_dir = root / ".mazu" / "skills"
        self.retention = retention

    def is_git_repo(self) -> bool:
        return (self.root / ".git").exists()

    def ensure_git_repo(self) -> None:
        if self.is_git_repo():
            return
        _git(self.root, ["init"])
        _git(self.root, ["add", "-A"])
        _git(self.root, ["commit", "-m", "Mazu: initial commit", "--allow-empty"])

    def is_dirty(self) -> bool:
        if not self.is_git_repo():
            return False
        result = _git(self.root, ["status", "--porcelain"])
        return bool(result.stdout.strip())

    def snapshot(self, messages: list[dict], trigger: str, summary: str = "") -> dict:
        self.ensure_git_repo()
        _git(self.root, ["add", "-A"])
        commit_msg = f"mazu checkpoint: {summary or trigger}"
        _git(self.root, ["commit", "-m", commit_msg, "--allow-empty"])
        commit_hash = _git(self.root, ["rev-parse", "HEAD"]).stdout.strip()

        entries = self.index.load()
        # Must be based on the highest id/step ever issued, not len(entries) -- once
        # pruning (below) removes old entries, len(entries) shrinks and would start
        # reissuing ids that collide with still-kept checkpoints, corrupting them.
        next_num = max((e["step"] for e in entries), default=0) + 1
        checkpoint_id = f"cp_{next_num:06d}"
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        (checkpoint_dir / "conversation.json").write_text(
            json.dumps(messages, indent=2), encoding="utf-8"
        )
        if self.memory_db_path.exists():
            _backup_sqlite(self.memory_db_path, checkpoint_dir / "memory.db")
        if self.skills_dir.exists():
            skills_snapshot_dir = checkpoint_dir / "skills"
            if skills_snapshot_dir.exists():  # defensive: guard against a stale/reused id
                shutil.rmtree(skills_snapshot_dir)
            shutil.copytree(self.skills_dir, skills_snapshot_dir)

        entry = {
            "id": checkpoint_id,
            "step": next_num,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit_hash,
            "trigger": trigger,
            "summary": summary or trigger,
        }
        self.index.append(entry)
        self.prune()
        return entry

    def prune(self, keep_last: int | None = None) -> int:
        """Deletes on-disk snapshot data (memory.db/skills/conversation.json copies)
        for all but the most recent `keep_last` checkpoints, and removes their index
        entries to match (a pruned checkpoint is no longer a valid rollback target —
        its git commit is still reachable via `git log`/`git checkout` manually, only
        our redundant bookkeeping copy is gone). Returns how many were pruned.
        """
        keep = keep_last if keep_last is not None else self.retention
        entries = self.index.load()
        if len(entries) <= keep:
            return 0
        to_prune = entries[:-keep] if keep > 0 else entries
        kept = entries[-keep:] if keep > 0 else []
        for entry in to_prune:
            checkpoint_dir = self.checkpoints_dir / entry["id"]
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
        self.index.save(kept)
        return len(to_prune)

    def list_checkpoints(self) -> list[dict]:
        return self.index.load()

    def preview_rollback(self, checkpoint_id: str | None = None) -> tuple[dict, str]:
        entry = self.index.get(checkpoint_id) if checkpoint_id else self.index.last()
        if entry is None:
            available = ", ".join(e["id"] for e in self.index.load()) or "(none)"
            raise ValueError(f"No checkpoint found for id={checkpoint_id!r}. Available: {available}")
        diff = _git(self.root, ["diff", entry["git_commit"], "HEAD", "--stat"]).stdout
        return entry, diff

    def restore(self, checkpoint_id: str) -> dict:
        entry = self.index.get(checkpoint_id)
        if entry is None:
            raise ValueError(f"No checkpoint found for id={checkpoint_id!r}")

        _git(self.root, ["reset", "--hard", entry["git_commit"]])
        # Removes untracked files created after this checkpoint. Respects .gitignore
        # by default, so .mazu/ (which mazu init/chat always gitignores) is untouched.
        _git(self.root, ["clean", "-fd"])

        checkpoint_dir = self.checkpoints_dir / entry["id"]
        snapshot_db = checkpoint_dir / "memory.db"
        if snapshot_db.exists():
            _backup_sqlite(snapshot_db, self.memory_db_path)

        # Skills live in .mazu/, which is gitignored, so `git clean -fd` above never
        # touches them — they need their own restore, mirroring the memory.db handling.
        if self.skills_dir.exists():
            shutil.rmtree(self.skills_dir)
        snapshot_skills = checkpoint_dir / "skills"
        if snapshot_skills.exists():
            shutil.copytree(snapshot_skills, self.skills_dir)

        conversation_path = checkpoint_dir / "conversation.json"
        messages = (
            json.loads(conversation_path.read_text(encoding="utf-8"))
            if conversation_path.exists()
            else []
        )

        self.index.truncate_after(entry["id"])
        return {"entry": entry, "messages": messages}
