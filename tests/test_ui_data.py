import subprocess
from pathlib import Path

import pytest

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.memory.store import MemoryStore
from mazu.ui.data import load_actions, load_checkpoints, load_memories, load_sessions


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# load_checkpoints
# ---------------------------------------------------------------------------


def test_load_checkpoints_empty(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    assert load_checkpoints(manager) == []


def test_load_checkpoints_newest_first(project: Path):
    manager = CheckpointManager(project)
    first = manager.snapshot(messages=[], trigger="manual", summary="first")
    second = manager.snapshot(messages=[], trigger="manual", summary="second")

    rows = load_checkpoints(manager)
    assert [r.id for r in rows] == [second["id"], first["id"]]


def test_load_checkpoints_includes_files_changed(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")
    (project / "a.py").write_text("x = 1")
    manager.snapshot(messages=[], trigger="manual")

    rows = load_checkpoints(manager)
    assert rows[0].files_changed == ["a.py"]  # newest first -- the one that added a.py


def test_load_checkpoints_fields_match_the_entry(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual", summary="a summary")

    row = load_checkpoints(manager)[0]
    assert row.id == entry["id"]
    assert row.trigger == "manual"
    assert row.summary == "a summary"


# ---------------------------------------------------------------------------
# load_memories
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


def test_load_memories_empty(memory_store: MemoryStore):
    assert load_memories(memory_store) == []


def test_load_memories_maps_fields(memory_store: MemoryStore):
    memory_store.add(
        category="decision", title="Use PostgreSQL", body="For concurrency", tags="db,postgres"
    )
    rows = load_memories(memory_store)
    assert len(rows) == 1
    assert rows[0].category == "decision"
    assert rows[0].title == "Use PostgreSQL"
    assert rows[0].body == "For concurrency"
    assert rows[0].tags == "db,postgres"
    assert rows[0].pinned is False


def test_load_memories_pinned_flag_is_a_real_bool(memory_store: MemoryStore):
    memory_store.add(category="fact", title="Pinned", body="x", pinned=True)
    rows = load_memories(memory_store)
    assert rows[0].pinned is True


def test_load_memories_excludes_superseded(memory_store: MemoryStore):
    old_id = memory_store.add(category="decision", title="Old", body="x")
    new_id = memory_store.add(category="decision", title="New", body="y")
    memory_store.supersede(old_id, new_id)

    rows = load_memories(memory_store)
    assert [r.title for r in rows] == ["New"]


# ---------------------------------------------------------------------------
# load_sessions / load_actions
# ---------------------------------------------------------------------------


@pytest.fixture
def action_log_store(tmp_path: Path) -> ActionLogStore:
    s = ActionLogStore(tmp_path / "action_log.db")
    yield s
    s.close()


def test_load_sessions_empty(action_log_store: ActionLogStore):
    assert load_sessions(action_log_store) == []


def test_load_sessions_aggregates_correctly(action_log_store: ActionLogStore):
    action_log_store.log("s1", "run", "write_file", "{}", "ok", "x", None)
    action_log_store.log("s1", "run", "write_file", "{}", "error", "y", None)

    rows = load_sessions(action_log_store)
    assert len(rows) == 1
    assert rows[0].session_id == "s1"
    assert rows[0].action_count == 2
    assert rows[0].error_count == 1


def test_load_sessions_respects_limit(action_log_store: ActionLogStore):
    for i in range(5):
        action_log_store.log(f"s{i}", "run", "read_file", "{}", "ok", "x", None)
    assert len(load_sessions(action_log_store, limit=2)) == 2


def test_load_actions_empty(action_log_store: ActionLogStore):
    assert load_actions(action_log_store, "nope") == []


def test_load_actions_maps_fields(action_log_store: ActionLogStore):
    action_log_store.log(
        "s1", "run", "write_file", '{"path": "a.py"}', "ok", "Wrote 5 bytes", "a.py"
    )
    rows = load_actions(action_log_store, "s1")
    assert len(rows) == 1
    assert rows[0].tool_name == "write_file"
    assert rows[0].outcome == "ok"
    assert rows[0].output_summary == "Wrote 5 bytes"
    assert rows[0].changed_file == "a.py"


def test_load_actions_only_returns_the_requested_session(action_log_store: ActionLogStore):
    action_log_store.log("s1", "run", "read_file", "{}", "ok", "s1 action", None)
    action_log_store.log("s2", "run", "read_file", "{}", "ok", "s2 action", None)

    rows = load_actions(action_log_store, "s1")
    assert len(rows) == 1
    assert rows[0].output_summary == "s1 action"
