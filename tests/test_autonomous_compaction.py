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
from mazu.tools.registry import ToolRegistry


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
