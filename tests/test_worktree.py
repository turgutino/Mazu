"""Tests for mazu/checkpoint/worktree.py -- the git worktree helpers `mazu explore`
uses to run N branches in real, isolated working directories in parallel (unlike
CheckpointManager.fork(), which does a real `git checkout` on the single self.root
working tree and can't be called N times concurrently).
"""

import subprocess
from pathlib import Path

import pytest

from mazu.checkpoint.worktree import add_worktree, list_worktrees, remove_worktree


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root)
    (root / "a.txt").write_text("hello")
    subprocess.run(["git", "add", "-A"], cwd=root)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root)
    return root


def _head_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()


def test_add_worktree_creates_a_real_working_directory(repo: Path, tmp_path: Path):
    commit = _head_commit(repo)
    wt = tmp_path / "wt1"

    result = add_worktree(repo, wt, "explore-branch-a", commit)

    assert result.returncode == 0
    assert wt.exists()
    assert (wt / "a.txt").exists()
    assert (wt / "a.txt").read_text() == "hello"


def test_add_worktree_creates_the_named_branch(repo: Path, tmp_path: Path):
    commit = _head_commit(repo)
    wt = tmp_path / "wt2"
    add_worktree(repo, wt, "explore-branch-b", commit)

    branches = subprocess.run(
        ["git", "branch", "--list", "explore-branch-b"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "explore-branch-b" in branches


def test_two_worktrees_are_independent(repo: Path, tmp_path: Path):
    """The whole reason mazu explore uses worktrees instead of sequential fork()
    calls: two branches must be able to have DIFFERENT file content checked out
    at the SAME time, in the SAME repo, without one clobbering the other.
    """
    commit = _head_commit(repo)
    wt_a = tmp_path / "wt_a"
    wt_b = tmp_path / "wt_b"
    add_worktree(repo, wt_a, "branch-a", commit)
    add_worktree(repo, wt_b, "branch-b", commit)

    (wt_a / "a.txt").write_text("changed by A")
    (wt_b / "a.txt").write_text("changed by B")

    assert (wt_a / "a.txt").read_text() == "changed by A"
    assert (wt_b / "a.txt").read_text() == "changed by B"
    # The origin repo's own working tree is untouched by either worktree's edits.
    assert (repo / "a.txt").read_text() == "hello"


def test_add_worktree_failure_returns_nonzero_not_an_exception(repo: Path, tmp_path: Path):
    """An invalid base commit should surface as a normal failed CompletedProcess
    (matching every other _git() call site in this codebase), not raise -- the
    caller (run_explore) is responsible for checking returncode and raising a
    clear error itself.
    """
    wt = tmp_path / "wt_bad"
    result = add_worktree(repo, wt, "branch-bad", "not-a-real-commit-hash")
    assert result.returncode != 0
    assert not wt.exists()


def test_remove_worktree_deletes_the_directory(repo: Path, tmp_path: Path):
    commit = _head_commit(repo)
    wt = tmp_path / "wt_remove"
    add_worktree(repo, wt, "branch-remove", commit)
    assert wt.exists()

    result = remove_worktree(repo, wt)

    assert result.returncode == 0
    assert not wt.exists()


def test_list_worktrees_includes_the_main_and_added_ones(repo: Path, tmp_path: Path):
    commit = _head_commit(repo)
    wt = tmp_path / "wt_list"
    add_worktree(repo, wt, "branch-list", commit)

    entries = list_worktrees(repo)
    paths = [e["path"] for e in entries]

    # Git normalizes path separators/casing on its own -- compare by suffix match
    # rather than exact string equality, which is what actually matters here.
    assert any(str(wt).replace("\\", "/") in p.replace("\\", "/") for p in paths)
    branches = [e["branch"] for e in entries]
    assert "branch-list" in branches


def test_list_worktrees_empty_repo_still_reports_main(repo: Path):
    entries = list_worktrees(repo)
    assert len(entries) >= 1
