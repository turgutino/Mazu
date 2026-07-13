"""Verifies Phase F's wiring in run_autonomous: RunStore tracks a run's config,
progress, and outcome; a resumed run reuses the same session_id without re-inserting
into MemoryStore's sessions table; and the structured end-of-run report reflects what
actually happened (files changed, checkpoints created, memories saved, tool errors).
"""

import subprocess
from pathlib import Path

import pytest

import mazu.agent.autonomous as autonomous_module
from mazu.action_log.store import ActionLogStore
from mazu.agent.autonomous import run_autonomous
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.types import AgentResponse
from mazu.memory.store import MemoryStore
from mazu.runs.store import RunStore
from mazu.tools.base import Tool, ToolResult
from mazu.tools.fs import make_fs_tools
from mazu.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def _tool_use_response(tool_name: str, tool_input: dict, tool_use_id: str = "t1") -> AgentResponse:
    return AgentResponse(
        stop_reason="tool_use",
        content=[{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": tool_input}],
        usage={},
    )


def _end_turn_response(text: str = "done") -> AgentResponse:
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": text}], usage={})


def _error_tool(name: str = "broken_tool") -> Tool:
    return Tool(
        name=name,
        description="always fails",
        input_schema={"type": "object"},
        handler=lambda inp: ToolResult("boom", is_error=True),
        destructive=False,
    )


# ---------------------------------------------------------------------------
# RunStore lifecycle wiring
# ---------------------------------------------------------------------------


def test_fresh_run_creates_a_run_row_and_marks_it_completed(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    run_store = RunStore(tmp_path / "runs.db")

    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        model="deepseek:deepseek-chat",
        run_store=run_store,
    )

    row = RunStore(tmp_path / "runs.db").get("r1")
    assert row["status"] == "completed"
    assert row["stop_reason"] == "end_turn"
    assert row["task"] == "do something"
    assert row["ended_at"] is not None


def test_run_stopped_by_max_steps_records_that_stop_reason(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    run_store = RunStore(tmp_path / "runs.db")

    responses = iter([_tool_use_response("read_file", {"path": "a.py"})] * 10)
    registry = ToolRegistry()
    registry.register(
        Tool("read_file", "reads", {"type": "object"}, lambda inp: ToolResult("x"), False)
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="loop forever",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=2,
        model="deepseek:deepseek-chat",
        run_store=run_store,
    )

    row = RunStore(tmp_path / "runs.db").get("r1")
    assert row["status"] == "stopped"
    assert row["stop_reason"] == "max_steps"


def test_run_store_tracks_checkpoints_created(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    run_store = RunStore(tmp_path / "runs.db")
    registry = ToolRegistry()
    for tool in make_fs_tools(tmp_path):
        registry.register(tool)

    responses = iter(
        [
            _tool_use_response("write_file", {"path": "a.py", "content": "x"}, "t1"),
            _tool_use_response("write_file", {"path": "b.py", "content": "y"}, "t2"),
            _end_turn_response(),
        ]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="write two files",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        checkpoint_every=1,
        model="deepseek:deepseek-chat",
        run_store=run_store,
    )

    row = RunStore(tmp_path / "runs.db").get("r1")
    assert row["checkpoints_created"] == 2
    assert row["last_checkpoint_id"] == "cp_000002"


def test_run_store_none_does_not_crash(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
        model="deepseek:deepseek-chat",
        run_store=None,
    )


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def test_resume_does_not_reinsert_into_memory_sessions_table(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    memory_store = MemoryStore(tmp_path / "memory.db")
    memory_store.start_session("r1")  # simulates the original (interrupted) run already having started it

    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())
    monkeypatch.setattr(
        autonomous_module, "finalize_session", lambda *a, **k: 0
    )  # avoid a real extraction call; not what this test is about

    # Must not raise sqlite3.IntegrityError from a duplicate PRIMARY KEY insert.
    run_autonomous(
        registry=ToolRegistry(),
        task="continue the task",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        memory_store=memory_store,
        max_steps=1,
        model="deepseek:deepseek-chat",
        resume_messages=[{"role": "user", "content": "original task"}],
    )


def test_resume_starts_from_the_provided_messages(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    captured_messages = []

    def _fake_run_turn(messages, *a, **k):
        captured_messages.append([m for m in messages])
        return _end_turn_response()

    monkeypatch.setattr(autonomous_module, "run_turn", _fake_run_turn)

    prior = [
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": [{"type": "text", "text": "working on it"}]},
    ]
    run_autonomous(
        registry=ToolRegistry(),
        task="original task",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
        model="deepseek:deepseek-chat",
        resume_messages=prior,
    )

    # The first run_turn call must have seen the resumed history, not a fresh
    # single-message conversation built from `task`.
    assert len(captured_messages[0]) == 2


def test_resume_does_not_re_register_run_in_run_store(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    run_store = RunStore(tmp_path / "runs.db")
    run_store.start("r1", "original task", "deepseek:deepseek-chat", 15, 1, False, None, None, False)
    run_store.update_progress("r1", 1, checkpoint_id="cp_000001")
    run_store.update_progress("r1", 2, checkpoint_id="cp_000002")
    run_store.update_progress("r1", 3, checkpoint_id="cp_000003")

    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    run_autonomous(
        registry=ToolRegistry(),
        task="original task",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
        model="deepseek:deepseek-chat",
        run_store=run_store,
        resume_messages=[{"role": "user", "content": "original task"}],
    )

    row = RunStore(tmp_path / "runs.db").get("r1")
    # start() must not have been called again (which would reset checkpoints_created
    # to 0 via a fresh INSERT) -- the prior progress must survive into this row, and
    # finish() must update the SAME row to reflect this continuation's outcome. The
    # fake run_turn ends the turn immediately (no tool_use round), so this resumed
    # continuation itself creates zero new checkpoints -- the count stays at the 3
    # set up above.
    assert row["checkpoints_created"] == 3
    assert row["status"] == "completed"
    assert row["stop_reason"] == "end_turn"


def test_dry_run_does_not_checkpoint_so_resume_finds_nothing(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    registry = ToolRegistry()
    for tool in make_fs_tools(tmp_path, dry_run=True):
        registry.register(tool)

    responses = iter(
        [_tool_use_response("write_file", {"path": "a.py", "content": "x"}), _end_turn_response()]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="write a.py",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        dry_run=True,
    )

    assert checkpoint_manager.latest_for_session("r1") is None


# ---------------------------------------------------------------------------
# end-of-run report
# ---------------------------------------------------------------------------


def test_report_lists_changed_files(tmp_path, monkeypatch, capsys):
    checkpoint_manager = CheckpointManager(tmp_path)
    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    registry = ToolRegistry()
    for tool in make_fs_tools(tmp_path):
        registry.register(tool)

    responses = iter(
        [
            _tool_use_response("write_file", {"path": "a.py", "content": "x"}, "t1"),
            _end_turn_response(),
        ]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="write a.py",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        model="deepseek:deepseek-chat",
        action_log_store=action_log_store,
    )

    out = capsys.readouterr().out
    assert "=== Run report ===" in out
    assert "a.py" in out
    assert "Stop reason: end_turn" in out


def test_report_counts_tool_errors(tmp_path, monkeypatch, capsys):
    checkpoint_manager = CheckpointManager(tmp_path)
    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    registry = ToolRegistry()
    registry.register(_error_tool())

    responses = iter([_tool_use_response("broken_tool", {}), _end_turn_response()])
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="try the broken tool",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        model="deepseek:deepseek-chat",
        action_log_store=action_log_store,
    )

    out = capsys.readouterr().out
    assert "Tool errors: 1" in out


def test_dry_run_report_labels_itself_as_a_preview(tmp_path, monkeypatch, capsys):
    checkpoint_manager = CheckpointManager(tmp_path)
    registry = ToolRegistry()
    for tool in make_fs_tools(tmp_path, dry_run=True):
        registry.register(tool)

    responses = iter(
        [_tool_use_response("write_file", {"path": "a.py", "content": "x"}), _end_turn_response()]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="write a.py",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "Dry-run report" in out
    assert "Files that would change" in out
    assert "a.py" in out


def test_report_reflects_memories_saved(tmp_path, monkeypatch, capsys):
    checkpoint_manager = CheckpointManager(tmp_path)
    memory_store = MemoryStore(tmp_path / "memory.db")

    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())
    monkeypatch.setattr(autonomous_module, "finalize_session", lambda *a, **k: 3)

    run_autonomous(
        registry=ToolRegistry(),
        task="do something memorable",
        session_id="r1",
        checkpoint_manager=checkpoint_manager,
        memory_store=memory_store,
        max_steps=1,
        model="deepseek:deepseek-chat",
    )

    out = capsys.readouterr().out
    assert "Memories saved: 3" in out
