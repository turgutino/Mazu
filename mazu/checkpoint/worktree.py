"""Git worktree lifecycle helpers for `mazu explore` (mazu/agent/explore.py).

Free functions taking `root` explicitly, mirroring checkpoint/manager.py's own
`_git(root, args)` style, rather than methods on CheckpointManager: worktree setup
happens BEFORE a per-branch CheckpointManager can exist for the new path (you can't
construct one rooted at a directory that doesn't exist yet), so this is inherently a
"pre-CheckpointManager" concern, not something that fits CheckpointManager's own
contract of "operates on an already-existing tree."
"""

import subprocess
from pathlib import Path


def _git(root: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def add_worktree(root: Path, worktree_path: Path, branch_name: str, base_commit: str) -> subprocess.CompletedProcess:
    """Creates a new branch `branch_name` at `base_commit` and checks it out into a
    separate working directory at `worktree_path` -- unlike CheckpointManager.fork(),
    this never touches `root`'s own working tree, since it's a different directory
    entirely. `worktree_path`'s parent must already exist; `git worktree add` creates
    the leaf directory itself.
    """
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    return _git(root, ["worktree", "add", "-b", branch_name, str(worktree_path), base_commit])


def remove_worktree(root: Path, worktree_path: Path, force: bool = False) -> subprocess.CompletedProcess:
    args = ["worktree", "remove", str(worktree_path)]
    if force:
        args.append("--force")
    return _git(root, args)


def list_worktrees(root: Path) -> list[dict]:
    """Parses `git worktree list --porcelain` into a list of
    {"path": ..., "branch": ..., "commit": ...} dicts. Blank lines separate entries;
    each entry has `worktree <path>`, `HEAD <commit>`, and `branch <ref>` lines
    (a detached worktree has no `branch` line, so it's reported as None).
    """
    result = _git(root, ["worktree", "list", "--porcelain"])
    entries: list[dict] = []
    current: dict = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line[len("worktree "):]
        elif line.startswith("HEAD "):
            current["commit"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            # e.g. "branch refs/heads/explore-abc123-anthropic-claude-sonnet-5"
            current["branch"] = line[len("branch "):].removeprefix("refs/heads/")
    if current:
        entries.append(current)
    for entry in entries:
        entry.setdefault("branch", None)
    return entries
