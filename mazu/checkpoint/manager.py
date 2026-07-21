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
    the conversation at that point. Rollback (restore()) is still a destructive,
    linear `git reset --hard` in place on whatever branch is current -- but
    checkpoints can also be forked (fork()) onto a new git branch non-destructively,
    with the checkpoint's memory/skills/conversation restored onto that branch too.
    `retention` bounds how many checkpoints' worth of memory.db/skills/
    conversation.json copies are kept on disk at once, per branch -- without this,
    `.mazu/checkpoints/` grows forever. Git history itself is never pruned, only our
    redundant snapshot copies.
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

    def _current_branch(self) -> str:
        return _git(self.root, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()

    def snapshot(
        self,
        messages: list[dict],
        trigger: str,
        summary: str = "",
        session_id: str | None = None,
        parent_checkpoint_id: str | None = None,
    ) -> dict:
        self.ensure_git_repo()
        _git(self.root, ["add", "-A"])
        commit_msg = f"mazu checkpoint: {summary or trigger}"
        _git(self.root, ["commit", "-m", commit_msg, "--allow-empty"])
        commit_hash = _git(self.root, ["rev-parse", "HEAD"]).stdout.strip()
        branch = self._current_branch()

        # parent_checkpoint_id is normally left to auto-resolve. A session_id'd
        # checkpoint (mazu run) points at that session's own last checkpoint -- the
        # real logical predecessor, not just "whatever the previous list entry
        # happens to be". This deliberately does NOT fall back further: a session_id
        # that has never checkpointed under itself yet (a fresh run, or the first
        # checkpoint of a fork) must stay a root unless fork() explicitly overrides
        # it via the parent_checkpoint_id param -- falling back to "whatever else was
        # last on this branch" here would wrongly link an unrelated run's history in.
        #
        # A checkpoint with NO session_id (manual `mazu checkpoint`, `mazu chat`) has
        # no session chain to consult at all, so it falls back to "the current
        # branch's own last checkpoint, from any session" -- this is what actually
        # reproduces the pre-branching behavior (diff against whatever came right
        # before it), since these calls are inherently sequential single-chain uses.
        if parent_checkpoint_id is None:
            if session_id is not None:
                parent_entry = self.latest_for_session(session_id)
            else:
                parent_entry = self.index.last_for_branch(branch)
            if parent_entry is not None:
                parent_checkpoint_id = parent_entry["id"]

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
            # Optional and additive -- older index entries simply lack this key
            # (.get("session_id") returns None for them). Lets a `mazu run` be
            # resumed from its own last checkpoint via latest_for_session() below,
            # without needing a separate session-to-checkpoint mapping file.
            "session_id": session_id,
            # Also optional/additive, same reasoning as session_id above. Sourced
            # from git itself (not a separate naming scheme) so it can never drift
            # from what `git branch`/`git log` actually show.
            "branch": branch,
            # None means "true root" -- either the very first checkpoint of the
            # project, or the first checkpoint of a session forked onto a fresh
            # branch via fork() with no history of its own yet.
            "parent_checkpoint_id": parent_checkpoint_id,
        }
        self.index.append(entry)
        self.prune()
        return entry

    def latest_for_session(self, session_id: str) -> dict | None:
        """The most recent checkpoint recorded under a given session/run id, or None
        if that session never checkpointed (e.g. a dry run, which never snapshots, or
        a session that ended before its first checkpoint-every boundary)."""
        matches = [e for e in self.index.load() if e.get("session_id") == session_id]
        if not matches:
            return None
        return max(matches, key=lambda e: e["step"])

    def prune(self, keep_last: int | None = None) -> int:
        """Deletes on-disk snapshot data (memory.db/skills/conversation.json copies)
        for all but the most recent `keep_last` checkpoints *per branch*, and removes
        their index entries to match (a pruned checkpoint is no longer a valid
        rollback target — its git commit is still reachable via `git log`/`git
        checkout` manually, only our redundant bookkeeping copy is gone). Returns how
        many were pruned.

        Grouped by branch (not one global suffix) so that a branch with few
        checkpoints doesn't get wiped out just because a different, more active
        branch produced many newer ones in the meantime — a global entries[-keep:]
        would otherwise happily prune away the only checkpoint a divergent branch
        has, purely because of chronological bad luck. Entries with no "branch" key
        (pre-branching history) are grouped together under None, same convention as
        CheckpointIndex.last_for_branch.
        """
        keep = keep_last if keep_last is not None else self.retention
        entries = self.index.load()

        by_branch: dict[str | None, list[dict]] = {}
        for entry in entries:
            by_branch.setdefault(entry.get("branch"), []).append(entry)

        to_prune: list[dict] = []
        for branch_entries in by_branch.values():
            if len(branch_entries) > keep:
                to_prune.extend(branch_entries[:-keep] if keep > 0 else branch_entries)

        if not to_prune:
            return 0
        # Recombine preserving original index order (not branch-grouped order) so
        # index.json's ordering/append semantics are otherwise unaffected.
        pruned_ids = {e["id"] for e in to_prune}
        kept_in_order = [e for e in entries if e["id"] not in pruned_ids]
        for entry in to_prune:
            checkpoint_dir = self.checkpoints_dir / entry["id"]
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
        self.index.save(kept_in_order)
        return len(to_prune)

    def list_checkpoints(self) -> list[dict]:
        return self.index.load()

    def _resolve_entry(self, checkpoint_id: str | None) -> dict:
        """Shared lookup for every method that takes an optional checkpoint id --
        None means "the most recent one on the current branch" (falling back to
        "most recent overall" only if the current branch has no checkpoints of its
        own yet, e.g. every pre-branching repo), matching how /rollback and `mazu
        rollback` already behave with no argument. Without the branch scoping, a
        no-argument `mazu rollback` on a feature branch could silently target a
        checkpoint that was actually made on a different, divergent branch just
        because it happened to be appended to the index later.
        """
        if checkpoint_id:
            entry = self.index.get(checkpoint_id)
        else:
            entry = self.index.last_for_branch(self._current_branch()) or self.index.last()
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

    def inspect_memory(self, checkpoint_id: str | None = None) -> list[dict]:
        """Reads the memories captured in a checkpoint's memory.db *snapshot*
        directly (not the live, current memory.db) -- what the project's memory
        actually looked like at that point in history. Empty list if no memory
        snapshot was captured for this checkpoint (e.g. .mazu/memory.db didn't
        exist yet when it was taken).
        """
        entry = self._resolve_entry(checkpoint_id)
        snapshot_db = self.checkpoints_dir / entry["id"] / "memory.db"
        if not snapshot_db.exists():
            return []
        from mazu.memory.store import MemoryStore

        store = MemoryStore(snapshot_db)
        try:
            return [dict(row) for row in store.all_active()]
        finally:
            store.close()

    def inspect_conversation(self, checkpoint_id: str | None = None) -> list[dict]:
        """Returns the raw message list captured in a checkpoint's
        conversation.json snapshot. Empty list if none was captured.
        """
        entry = self._resolve_entry(checkpoint_id)
        conversation_path = self.checkpoints_dir / entry["id"] / "conversation.json"
        if not conversation_path.exists():
            return []
        try:
            return json.loads(conversation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def compare(self, checkpoint_id_a: str, checkpoint_id_b: str) -> tuple[dict, dict, str]:
        """Diff between two checkpoints' commits directly (not either one vs. the
        current working tree, unlike diff_against_current) -- both refs are real
        commits here, so no working-tree-vs-HEAD subtlety applies.
        """
        entry_a = self._resolve_entry(checkpoint_id_a)
        entry_b = self._resolve_entry(checkpoint_id_b)
        diff = self._diff_stat(entry_a["git_commit"], entry_b["git_commit"])
        return entry_a, entry_b, diff

    def branch_from(self, checkpoint_id: str | None, branch_name: str) -> dict:
        """Creates a new git branch pointing at a checkpoint's commit, without
        touching the current branch or working tree -- unlike restore(), this
        never runs `git reset`/`git clean` and never touches memory.db or skills.
        The current branch stays checked out; the new branch is just a pointer the
        user can `git checkout` into themselves when ready. Deliberately
        lightweight and git-only: the point is a cheap way to explore an alternate
        path from history without a full, stateful rollback.
        """
        entry = self._resolve_entry(checkpoint_id)
        result = _git(self.root, ["branch", branch_name, entry["git_commit"]])
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or f"Failed to create branch {branch_name!r}")
        return entry

    def fork(self, checkpoint_id: str | None, branch_name: str) -> dict:
        """The stateful counterpart to branch_from(): creates the branch pointer
        (reusing branch_from() as-is), checks it out, and restores memory.db/skills
        onto it -- the same restore logic restore() already uses, at the same
        fidelity, so a forked branch's working state actually matches the
        checkpoint it forked from, not just its git commit.

        Deliberately never calls index.truncate_after() (unlike restore()): forking
        is additive divergence, not a rollback. The origin branch's later
        checkpoints must stay exactly as valid as they were before the fork --
        truncating them here would silently destroy history restore() has no
        business touching. This is the one property that makes fork() safe to use
        as a "try something different" operation instead of a destructive one.
        """
        entry = self.branch_from(checkpoint_id, branch_name)
        result = _git(self.root, ["checkout", branch_name])
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or f"Failed to check out branch {branch_name!r}")

        checkpoint_dir = self.checkpoints_dir / entry["id"]
        snapshot_db = checkpoint_dir / "memory.db"
        if snapshot_db.exists():
            _backup_sqlite(snapshot_db, self.memory_db_path)

        if self.skills_dir.exists():
            shutil.rmtree(self.skills_dir)
        snapshot_skills = checkpoint_dir / "skills"
        if snapshot_skills.exists():
            shutil.copytree(snapshot_skills, self.skills_dir)

        messages = self.inspect_conversation(entry["id"])
        return {"entry": entry, "messages": messages}

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
        """Every checkpoint's index metadata enriched with what changed since its
        *logical parent* (via parent_checkpoint_id -- its actual git ancestor, not
        just whatever entry happens to sit before it in the flat index list, which
        stops being the same thing the moment two branches' checkpoints interleave
        chronologically in one list) and whether a memory/skills snapshot exists.
        A root checkpoint (parent_checkpoint_id is None, or points at an entry that
        has since been pruned out of the index) has no predecessor to diff against,
        so its files_changed is empty rather than guessed at. For pre-branching
        entries that never had parent_checkpoint_id recorded, this falls back to the
        previous list entry -- identical to the old behavior, so output for today's
        single-chain case is unchanged.
        """
        entries = self.index.load()
        by_id = {e["id"]: e for e in entries}
        result = []
        prev_commit = None
        for entry in entries:
            if "parent_checkpoint_id" in entry:
                parent_entry = by_id.get(entry.get("parent_checkpoint_id"))
                parent_commit = parent_entry["git_commit"] if parent_entry else None
            else:
                parent_commit = prev_commit
            files_changed = (
                self._diff_names(parent_commit, entry["git_commit"]) if parent_commit is not None else []
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
