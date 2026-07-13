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

    def _resolve_entry(self, checkpoint_id: str | None) -> dict:
        """Shared lookup for every method that takes an optional checkpoint id --
        None means "the most recent one", matching how /rollback and `mazu
        rollback` already behave with no argument.
        """
        entry = self.index.get(checkpoint_id) if checkpoint_id else self.index.last()
        if entry is None:
            available = ", ".join(e["id"] for e in self.index.load()) or "(none)"
            raise ValueError(f"No checkpoint found for id={checkpoint_id!r}. Available: {available}")
        return entry

    def _diff_args(self, commit_hash: str, against: str) -> list[str]:
        # "HEAD" is the sentinel for "the live working tree, including uncommitted
        # changes" -- `git diff <commit> HEAD` (two explicit refs) compares two
        # *commits* and would silently show nothing for any edit that hasn't been
        # committed yet (a very real case: this checkpoint IS the current HEAD, and
        # the working tree has since been hand-edited). Omitting the second ref
        # entirely is git's own way of diffing a commit against the working tree.
        # A real commit hash (used by timeline_entries for checkpoint-to-checkpoint
        # comparisons) is passed through as an explicit second ref as normal.
        return ["diff", commit_hash] if against == "HEAD" else ["diff", commit_hash, against]

    def _diff_stat(self, commit_hash: str, against: str = "HEAD") -> str:
        return _git(self.root, [*self._diff_args(commit_hash, against), "--stat"]).stdout

    def _diff_names(self, commit_hash: str, against: str) -> list[str]:
        result = _git(self.root, [*self._diff_args(commit_hash, against), "--name-only"])
        names = [line for line in result.stdout.splitlines() if line.strip()]
        if against == "HEAD":
            # `git diff` never lists untracked files regardless of which ref it's
            # compared against -- a file created since the checkpoint but never
            # `git add`ed would otherwise silently vanish from "what changed",
            # which defeats the point of a diff view. Only relevant when comparing
            # against the live working tree ("HEAD"); a commit-to-commit diff (used
            # by timeline_entries) is always clean, since snapshot() always `git
            # add -A`s before committing.
            names.extend(f for f in self._untracked_files() if f not in names)
        return names

    def _untracked_files(self) -> list[str]:
        result = _git(self.root, ["status", "--porcelain"])
        return [line[3:] for line in result.stdout.splitlines() if line.startswith("??")]

    def has_memory_snapshot(self, checkpoint_id: str) -> bool:
        return (self.checkpoints_dir / checkpoint_id / "memory.db").exists()

    def has_skills_snapshot(self, checkpoint_id: str) -> bool:
        return (self.checkpoints_dir / checkpoint_id / "skills").exists()

    def preview_rollback(self, checkpoint_id: str | None = None) -> tuple[dict, str]:
        entry = self._resolve_entry(checkpoint_id)
        diff = self._diff_stat(entry["git_commit"])
        return entry, diff

    def diff_against_current(self, checkpoint_id: str | None = None) -> tuple[dict, str]:
        """Like preview_rollback, but meant for inspection (`mazu checkpoint diff`),
        not as a precursor to an actual rollback. Unlike preview_rollback's raw
        `--stat` output (kept as-is to avoid touching the existing rollback
        confirmation flow), this also calls out untracked new files explicitly --
        `git diff --stat` alone silently omits them entirely (see _diff_names).
        """
        entry = self._resolve_entry(checkpoint_id)
        diff = self._diff_stat(entry["git_commit"])
        untracked = self._untracked_files()
        if untracked:
            diff = diff.rstrip() + "\n\nNew (untracked) files:\n" + "\n".join(f"  {f}" for f in untracked)
        return entry, diff

    def show_entry(self, checkpoint_id: str | None = None) -> dict:
        """Full detail for one checkpoint: its index metadata plus how many messages
        its conversation snapshot holds and whether memory/skills were captured.
        """
        entry = self._resolve_entry(checkpoint_id)
        conversation_path = self.checkpoints_dir / entry["id"] / "conversation.json"
        message_count = 0
        if conversation_path.exists():
            try:
                message_count = len(json.loads(conversation_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                message_count = 0
        return {
            **entry,
            "message_count": message_count,
            "has_memory_snapshot": self.has_memory_snapshot(entry["id"]),
            "has_skills_snapshot": self.has_skills_snapshot(entry["id"]),
        }

    def timeline_entries(self) -> list[dict]:
        """Every checkpoint's index metadata enriched with what changed since the
        *previous* checkpoint (not since HEAD -- this is a step-by-step history
        view, not a series of cumulative diffs) and whether a memory/skills
        snapshot exists. The first checkpoint has no predecessor to diff against,
        so its files_changed is empty rather than guessed at.
        """
        entries = self.index.load()
        result = []
        prev_commit = None
        for entry in entries:
            files_changed = (
                self._diff_names(prev_commit, entry["git_commit"]) if prev_commit is not None else []
            )
            result.append(
                {
                    **entry,
                    "files_changed": files_changed,
                    "has_memory_snapshot": self.has_memory_snapshot(entry["id"]),
                    "has_skills_snapshot": self.has_skills_snapshot(entry["id"]),
                }
            )
            prev_commit = entry["git_commit"]
        return result

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
