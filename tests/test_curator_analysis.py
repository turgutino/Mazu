import subprocess
from pathlib import Path

import pytest

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.curator.analysis.branches import find_abandoned_branches
from mazu.curator.analysis.patterns import find_common_tool_sequences
from mazu.diagnostics import ensure_gitignore


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    ensure_gitignore(root)  # keeps .mazu/ (including checkpoints' own index.json) out of git,
    # matching real `mazu init` usage -- without this, snapshot()'s own `git add -A`
    # commits index.json itself, and switching branches later then conflicts with
    # its own uncommitted local state.
    (root / "f.txt").write_text("hello", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")


@pytest.fixture
def repo(tmp_path):
    _init_repo(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# find_abandoned_branches
# ---------------------------------------------------------------------------


def test_no_checkpoints_yields_no_candidates(repo):
    assert find_abandoned_branches(repo) == []


def test_current_branch_is_never_flagged(repo):
    manager = CheckpointManager(repo)
    manager.snapshot(messages=[], trigger="manual", summary="baseline")
    assert find_abandoned_branches(repo) == []


def test_old_forked_branch_is_flagged(repo):
    manager = CheckpointManager(repo)
    entry = manager.snapshot(messages=[], trigger="manual", summary="baseline")
    manager.fork(entry["id"], "side-quest")
    # fork() alone only switches branches -- it doesn't record a new checkpoint
    # index entry. A real snapshot() call on the new branch is needed both to (a)
    # produce an index entry tagged branch='side-quest' for find_abandoned_branches
    # to group by, and (b) create a real divergent commit (fork() alone leaves
    # side-quest pointing at the exact same commit as the origin branch, which
    # 'git branch --merged' correctly treats as trivially merged).
    (repo / "side.txt").write_text("side work", encoding="utf-8")
    manager.snapshot(messages=[], trigger="manual", summary="side work")
    default_branch = "master" if _default_branch_is_master(repo) else "main"
    _git(repo, "checkout", "-q", default_branch)

    candidates = find_abandoned_branches(repo, min_age_days=0)
    branch_names = {c["branch"] for c in candidates}
    assert "side-quest" in branch_names


def _default_branch_is_master(repo: Path) -> bool:
    proc = subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True)
    # After checking out side-quest we can't tell directly; check reflog/log instead.
    proc2 = subprocess.run(["git", "branch", "--list"], cwd=repo, capture_output=True, text=True)
    return "master" in proc2.stdout


def test_recent_branch_not_flagged_with_default_age_threshold(repo):
    manager = CheckpointManager(repo)
    entry = manager.snapshot(messages=[], trigger="manual", summary="baseline")
    manager.fork(entry["id"], "side-quest")
    (repo / "side.txt").write_text("side work", encoding="utf-8")
    manager.snapshot(messages=[], trigger="manual", summary="side work")
    branch = "master" if _default_branch_is_master(repo) else "main"
    _git(repo, "checkout", "-q", branch)

    # Default ABANDONED_AGE_DAYS=14 -- a branch created moments ago must not be flagged.
    candidates = find_abandoned_branches(repo)
    assert candidates == []


def test_deleted_branch_is_not_flagged(repo):
    """A branch that no longer exists has nothing left for a human to act on."""
    manager = CheckpointManager(repo)
    entry = manager.snapshot(messages=[], trigger="manual", summary="baseline")
    manager.fork(entry["id"], "side-quest")
    (repo / "side.txt").write_text("side work", encoding="utf-8")
    manager.snapshot(messages=[], trigger="manual", summary="side work")
    branch = "master" if _default_branch_is_master(repo) else "main"
    _git(repo, "checkout", "-q", branch)
    _git(repo, "branch", "-D", "side-quest")

    candidates = find_abandoned_branches(repo, min_age_days=0)
    assert candidates == []


# ---------------------------------------------------------------------------
# find_common_tool_sequences
# ---------------------------------------------------------------------------


@pytest.fixture
def action_log(tmp_path):
    s = ActionLogStore(tmp_path / "action_log.db")
    yield s
    s.close()


def test_no_sessions_yields_no_patterns(action_log):
    assert find_common_tool_sequences(action_log) == []


def test_repeated_sequence_is_found(action_log):
    for session_id in ("s1", "s2", "s3"):
        action_log.log(session_id, "chat", "read_file", "{}", "ok", "x", None)
        action_log.log(session_id, "chat", "run_shell", "{}", "ok", "x", None)

    patterns = find_common_tool_sequences(action_log, min_count=3)
    assert len(patterns) == 1
    assert patterns[0]["sequence"] == ["read_file", "run_shell"]
    assert patterns[0]["count"] == 3


def test_below_min_count_is_excluded(action_log):
    action_log.log("s1", "chat", "read_file", "{}", "ok", "x", None)
    action_log.log("s1", "chat", "run_shell", "{}", "ok", "x", None)

    assert find_common_tool_sequences(action_log, min_count=3) == []
