import subprocess
from pathlib import Path

import pytest

from mazu.checkpoint.manager import CheckpointManager


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    # CI runners (and some dev machines) have no global git user configured;
    # `git commit` fails without one. Scope it to this test's HOME so it never
    # touches the real global git config.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    # Mirrors cli.py's _ensure_gitignore(), which every real command calls before
    # constructing a CheckpointManager. Without it, `.mazu/` (checkpoints + skills)
    # would be tracked/cleaned by git like any other file, which the restore logic
    # doesn't expect -- this fixture keeps the test faithful to actual usage.
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    return tmp_path


def test_snapshot_creates_checkpoint_and_commit(project: Path):
    (project / "a.py").write_text("print('a')")
    manager = CheckpointManager(project)

    entry = manager.snapshot(messages=[{"role": "user", "content": "hi"}], trigger="manual")

    assert entry["id"] == "cp_000001"
    assert (manager.checkpoints_dir / entry["id"] / "conversation.json").exists()


def test_rollback_restores_deleted_file(project: Path):
    (project / "a.py").write_text("print('a')")
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    (project / "a.py").unlink()
    (project / "b.py").write_text("print('b')")

    manager.restore(entry["id"])

    assert (project / "a.py").exists()
    assert not (project / "b.py").exists()


def test_rollback_restores_memory_db(project: Path):
    from mazu.memory.store import MemoryStore

    mazu_dir = project / ".mazu"
    mazu_dir.mkdir()
    store = MemoryStore(mazu_dir / "memory.db")
    store.add(category="fact", title="Before checkpoint", body="x")
    store.close()

    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    store = MemoryStore(mazu_dir / "memory.db")
    store.add(category="fact", title="After checkpoint", body="y")
    store.close()

    manager.restore(entry["id"])

    store = MemoryStore(mazu_dir / "memory.db")
    titles = {row["title"] for row in store.all_active()}
    store.close()
    assert titles == {"Before checkpoint"}


def test_rollback_restores_skills(project: Path):
    from mazu.skills.manager import SkillManager

    skills = SkillManager(project)
    skills.save("skill_one", "does a thing", "def run(args):\n    return 'ok'")

    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    skills.save("skill_two", "does another thing", "def run(args):\n    return 'ok2'")
    assert len(skills.list()) == 2

    manager.restore(entry["id"])

    assert [m["name"] for m in skills.list()] == ["skill_one"]


def test_checkpoint_ids_survive_pruning_without_collision(project: Path):
    manager = CheckpointManager(project, retention=3)
    entries = [manager.snapshot(messages=[], trigger="manual") for _ in range(5)]

    # Pruning already ran inline (snapshot calls prune()); only the last 3 remain on disk.
    remaining = manager.list_checkpoints()
    assert len(remaining) == 3
    assert [e["id"] for e in remaining] == [e["id"] for e in entries[-3:]]

    # A new checkpoint must not reuse an id that collides with history.
    new_entry = manager.snapshot(messages=[], trigger="manual")
    assert new_entry["id"] == "cp_000006"


def test_prune_keeps_only_most_recent(project: Path):
    manager = CheckpointManager(project, retention=100)
    for _ in range(5):
        manager.snapshot(messages=[], trigger="manual")

    pruned_count = manager.prune(keep_last=2)

    assert pruned_count == 3
    assert len(manager.list_checkpoints()) == 2


def test_restore_unknown_checkpoint_raises(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    with pytest.raises(ValueError):
        manager.restore("cp_999999")


def test_is_dirty_detects_uncommitted_changes(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    assert manager.is_dirty() is False

    (project / "new_file.py").write_text("x")
    assert manager.is_dirty() is True
