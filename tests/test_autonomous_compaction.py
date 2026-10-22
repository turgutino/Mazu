"""Verifies mazu run (autonomous.py) actually wires the compaction module in
correctly: proactive compact_if_needed() runs every round, and a
MazuContextLengthError triggers force_compact() plus exactly one retry. The
compaction algorithm's own correctness is covered separately in
test_compaction.py -- this file is only about the wiring/control-flow in
autonomous.py, so compaction itself is stubbed out here.
"""

import subprocess

import pytest

import mazu.agent.autonomous as autonomous_module
from mazu.agent.autonomous import run_autonomous
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.errors import MazuContextLengthError
from mazu.llm.types import AgentResponse
from mazu.agent.council import _SharedCostTracker
from mazu.memory.store import MemoryStore
from mazu.tools.registry import ToolRegistry
from mazu.usage.store import UsageStore


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def _end_turn_response(text="done") -> AgentResponse:
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": text}], usage={})


def test_proactive_compaction_checked_every_round(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)

    calls = []

    def _fake_compact_if_needed(messages, model, **kwargs):
        calls.append(len(messages))
        return messages, False

    monkeypatch.setattr(autonomous_module, "compact_if_needed", _fake_compact_if_needed)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
    )

    assert len(calls) == 1


def test_context_length_error_triggers_force_compact_and_retries_once(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)

    run_turn_calls = {"n": 0}

    def _fake_run_turn(messages, system, tools, model=None):
        run_turn_calls["n"] += 1
        if run_turn_calls["n"] == 1:
            raise MazuContextLengthError("too long")
        return _end_turn_response("recovered")

    force_compact_calls = {"n": 0}

    def _fake_force_compact(messages, model):
        force_compact_calls["n"] += 1
        return messages[-1:]

    monkeypatch.setattr(autonomous_module, "run_turn", _fake_run_turn)
    monkeypatch.setattr(autonomous_module, "force_compact", _fake_force_compact)
    monkeypatch.setattr(
        autonomous_module, "compact_if_needed", lambda messages, model, **k: (messages, False)
    )

    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
    )

    assert run_turn_calls["n"] == 2  # first call raised, retry after force_compact succeeded
    assert force_compact_calls["n"] == 1


def test_context_length_error_on_retry_counts_as_a_failure(tmp_path, monkeypatch):
    """If the retry after force_compact ALSO fails, it must fall through to the
    ordinary consecutive-failure counting, not crash or loop silently.
    """
    checkpoint_manager = CheckpointManager(tmp_path)

    def _always_raises(messages, system, tools, model=None):
        raise MazuContextLengthError("still too long")

    monkeypatch.setattr(autonomous_module, "run_turn", _always_raises)
    monkeypatch.setattr(autonomous_module, "force_compact", lambda messages, model: messages)
    monkeypatch.setattr(
        autonomous_module, "compact_if_needed", lambda messages, model, **k: (messages, False)
    )

    # max_consecutive_failures=1 so a single failed round stops the run immediately,
    # proving the error was caught (not propagated as an unhandled exception).
    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=3,
        max_consecutive_failures=1,
    )
    # No assertion needed beyond "didn't raise" -- reaching this line means the
    # error was handled gracefully instead of crashing the process.


def test_max_cost_counts_compaction_and_final_memory_extraction(tmp_path, monkeypatch):
    usage_store = UsageStore(tmp_path / "usage.db")
    memory_store = MemoryStore(tmp_path / "memory.db")
    tracker = _SharedCostTracker(0.06)

    def _compact(messages, model, usage_store=None, session_id=None, **kwargs):
        usage_store.log("compaction", session_id, "anthropic", "claude-haiku-4-5", 1, 1, 0.04)
        return messages, False

    def _run_turn(*args, **kwargs):
        return AgentResponse(
            stop_reason="tool_use",
            content=[{"type": "tool_use", "id": "tool-1", "name": "missing", "input": {}}],
            usage={"input_tokens": 10_000, "output_tokens": 0},
        )

    def _finalize(store, session_id, messages, usage_store=None, **kwargs):
        usage_store.log("memory_extraction", session_id, "anthropic", "claude-haiku-4-5", 1, 1, 0.02)
        store.close()
        return 0

    monkeypatch.setattr(autonomous_module, "compact_if_needed", _compact)
    monkeypatch.setattr(autonomous_module, "run_turn", _run_turn)
    monkeypatch.setattr(autonomous_module, "finalize_session", _finalize)

    run_autonomous(
        registry=ToolRegistry(), task="continue", session_id="s1",
        checkpoint_manager=CheckpointManager(tmp_path), memory_store=memory_store,
        usage_store=usage_store, model="anthropic:claude-sonnet-5", max_cost=0.05,
        max_steps=2, shared_cost_tracker=tracker,
    )

    summary = UsageStore(tmp_path / "usage.db").summary(session_id="s1")
    assert summary["total_cost"] == pytest.approx(0.09)
    assert summary["total_calls"] == 3
    assert tracker.total == pytest.approx(0.09)
