"""Tests for the Textual TUI (mazu/ui/app.py) using Textual's own Pilot-driven test
harness (App.run_test()) -- these run the real app in headless mode and simulate
real key presses / clicks, not mocks of Textual itself. What IS real underneath:
every mutation (rollback, pin/unpin) goes through the actual CheckpointManager/
MemoryStore against a real temp project with real git and real SQLite -- these tests
prove the UI layer wires those real operations correctly, the same way
test_ui_data.py already proves the data-loading layer is correct on its own.
"""

import subprocess
from pathlib import Path

import pytest
from textual.widgets import DataTable

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.memory.store import MemoryStore
from mazu.ui.app import MazuApp


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
# mounting / data population
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_mounts_and_populates_empty_tables(project: Path):
    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        checkpoint_table = app.query_one("#checkpoint_table", DataTable)
        memory_table = app.query_one("#memory_table", DataTable)
        session_table = app.query_one("#session_table", DataTable)
        assert checkpoint_table.row_count == 0
        assert memory_table.row_count == 0
        assert session_table.row_count == 0


@pytest.mark.asyncio
async def test_app_populates_checkpoint_table(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    manager.snapshot(messages=[], trigger="manual", summary="first checkpoint")

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#checkpoint_table", DataTable)
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_app_populates_memory_table(project: Path):
    store = MemoryStore(project / ".mazu" / "memory.db")
    store.add(category="decision", title="Use PostgreSQL", body="for concurrency")
    store.close()

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#memory_table", DataTable)
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_app_populates_session_table(project: Path):
    store = ActionLogStore(project / ".mazu" / "action_log.db")
    store.log("s1", "run", "write_file", "{}", "ok", "did something", None)
    store.close()

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#session_table", DataTable)
        assert table.row_count == 1


# ---------------------------------------------------------------------------
# rollback (real CheckpointManager.restore, real git/filesystem)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollback_confirmed_actually_restores_the_checkpoint(project: Path):
    manager = CheckpointManager(project)
    (project / "a.py").write_text("original")
    good = manager.snapshot(messages=[], trigger="manual", summary="good state")
    (project / "a.py").write_text("broken")
    manager.snapshot(messages=[], trigger="manual", summary="broken state")

    assert (project / "a.py").read_text() == "broken"

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#checkpoint_table", DataTable)
        table.focus()
        table.move_cursor(row=1)  # newest-first list -- row 1 is the older "good" checkpoint
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
        await pilot.click("#confirm")
        await pilot.pause()

    assert (project / "a.py").read_text() == "original"


@pytest.mark.asyncio
async def test_rollback_cancelled_does_not_restore(project: Path):
    manager = CheckpointManager(project)
    (project / "a.py").write_text("original")
    manager.snapshot(messages=[], trigger="manual", summary="good state")
    (project / "a.py").write_text("broken")
    manager.snapshot(messages=[], trigger="manual", summary="broken state")

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#checkpoint_table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
        await pilot.click("#cancel")
        await pilot.pause()

    assert (project / "a.py").read_text() == "broken"


@pytest.mark.asyncio
async def test_rollback_key_does_nothing_when_checkpoint_table_not_focused(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        memory_table = app.query_one("#memory_table", DataTable)
        memory_table.focus()
        await pilot.pause()
        # Must not raise or open a confirm modal -- "r" is scoped to the checkpoint
        # table specifically, and nothing is focused there right now.
        await pilot.press("r")
        await pilot.pause()
        assert len(app.screen_stack) == 1


# ---------------------------------------------------------------------------
# pin / unpin (real MemoryStore mutation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toggle_pin_pins_an_unpinned_memory(project: Path):
    store = MemoryStore(project / ".mazu" / "memory.db")
    memory_id = store.add(category="fact", title="A", body="a")
    store.close()

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#memory_table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("p")
        await pilot.pause()

    verify_store = MemoryStore(project / ".mazu" / "memory.db")
    assert verify_store.get(memory_id)["pinned"] == 1
    verify_store.close()


@pytest.mark.asyncio
async def test_toggle_pin_unpins_a_pinned_memory(project: Path):
    store = MemoryStore(project / ".mazu" / "memory.db")
    memory_id = store.add(category="fact", title="A", body="a", pinned=True)
    store.close()

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#memory_table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("p")
        await pilot.pause()

    verify_store = MemoryStore(project / ".mazu" / "memory.db")
    assert verify_store.get(memory_id)["pinned"] == 0
    verify_store.close()


# ---------------------------------------------------------------------------
# action log drill-down
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selecting_a_session_loads_its_actions(project: Path):
    store = ActionLogStore(project / ".mazu" / "action_log.db")
    store.log("s1", "run", "write_file", '{"path": "a.py"}', "ok", "Wrote 5 bytes", "a.py")
    store.log("s1", "run", "read_file", '{"path": "b.py"}', "error", "File not found", None)
    store.close()

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        session_table = app.query_one("#session_table", DataTable)
        action_table = app.query_one("#action_table", DataTable)
        assert action_table.row_count == 0

        session_table.focus()
        session_table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert action_table.row_count == 2


@pytest.mark.asyncio
async def test_selecting_different_sessions_replaces_action_table_not_appends(project: Path):
    store = ActionLogStore(project / ".mazu" / "action_log.db")
    store.log("s1", "run", "write_file", "{}", "ok", "s1 action", None)
    store.log("s2", "run", "write_file", "{}", "ok", "s2 action a", None)
    store.log("s2", "run", "write_file", "{}", "ok", "s2 action b", None)
    store.close()

    app = MazuApp(project)
    async with app.run_test() as pilot:
        await pilot.pause()
        session_table = app.query_one("#session_table", DataTable)
        action_table = app.query_one("#action_table", DataTable)

        # Sessions are listed most-recent-first -- s2 (2 actions) comes before s1.
        session_table.focus()
        session_table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert action_table.row_count == 2

        session_table.move_cursor(row=1)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert action_table.row_count == 1
