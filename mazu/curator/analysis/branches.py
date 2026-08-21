import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mazu.checkpoint.store import CheckpointIndex

ABANDONED_AGE_DAYS = 14


def _git_branches(root: Path) -> tuple[str | None, set[str], set[str]]:
    """Returns (current_branch, all_branches, merged_branches). Best-effort -- any
    git failure (not a repo, git not installed, a shallow clone with no branch
    info) yields empty/None rather than raising, since this is read-only analysis
    Curator uses as one input among several, not a hard requirement."""

    def _run(args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
        except Exception:
            return None

    current = None
    proc = _run(["branch", "--show-current"])
    if proc is not None and proc.returncode == 0:
        current = proc.stdout.strip() or None

    all_branches: set[str] = set()
    proc = _run(["branch", "--list", "--format=%(refname:short)"])
    if proc is not None and proc.returncode == 0:
        all_branches = {b.strip() for b in proc.stdout.splitlines() if b.strip()}

    merged: set[str] = set()
    proc = _run(["branch", "--merged", "--format=%(refname:short)"])
    if proc is not None and proc.returncode == 0:
        merged = {b.strip() for b in proc.stdout.splitlines() if b.strip()}

    return current, all_branches, merged


def find_abandoned_branches(root: Path, min_age_days: int = ABANDONED_AGE_DAYS) -> list[dict]:
    """Branches with real checkpoint history that nobody has returned to: not the
    current branch, not merged into it, still exist as a real git branch (a
    deleted branch has nothing left to act on), and whose most recent checkpoint
    is at least min_age_days old. Purely read-only/advisory -- Curator has no tool
    that deletes or merges a branch, this only ever surfaces the finding for
    curator_note or a human to act on.
    """
    index = CheckpointIndex(root / ".mazu" / "checkpoints")
    entries = index.load()
    if not entries:
        return []

    current, all_branches, merged = _git_branches(root)
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)

    by_branch: dict[str, list[dict]] = {}
    for e in entries:
        branch = e.get("branch") or "(unknown)"
        by_branch.setdefault(branch, []).append(e)

    candidates = []
    for branch, branch_entries in by_branch.items():
        if branch == current or branch in merged:
            continue
        if all_branches and branch not in all_branches:
            continue
        last_entry = max(branch_entries, key=lambda e: e["created_at"])
        try:
            last_dt = datetime.fromisoformat(last_entry["created_at"])
        except ValueError:
            continue
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        if last_dt < cutoff:
            candidates.append({
                "branch": branch,
                "checkpoint_count": len(branch_entries),
                "last_checkpoint_id": last_entry["id"],
                "last_checkpoint_at": last_entry["created_at"],
            })
    return candidates
