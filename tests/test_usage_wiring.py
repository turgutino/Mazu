"""Verifies mazu chat / mazu run / mazu council all correctly log to UsageStore, and
in particular that mazu council -- which runs member calls in parallel worker
threads via ThreadPoolExecutor -- never touches the UsageStore's sqlite3 connection
from more than one thread. sqlite3 connections are not thread-safe by default, so a
naive "log from inside the worker" implementation would either crash or silently
corrupt data; this uses a REAL ThreadPoolExecutor (not mocked) with several members
to actually exercise that risk, not just assert intent.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import mazu.agent.autonomous as autonomous_module
import mazu.agent.loop as loop_module
from mazu.agent.council import run_council
from mazu.agent.loop import run_chat_loop
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.types import AgentResponse
from mazu.tools.registry import ToolRegistry
from mazu.usage.store import UsageStore


def _end_turn_response(text: str = "done") -> AgentResponse:
    return AgentResponse(
        stop_reason="end_turn",
        content=[{"type": "text", "text": text}],
        usage={"input_tokens": 100, "output_tokens": 50},
    )


# ---------------------------------------------------------------------------
# mazu chat
# ---------------------------------------------------------------------------


def test_chat_logs_usage_and_prints_cost(tmp_path, monkeypatch, capsys):
    usage_store = UsageStore(tmp_path / "usage.db")

    def _fake_stream(messages, system, tools, on_delta, model=None):
        on_delta("hi there")
        return _end_turn_response("hi there")

    inputs = iter(["hello", ""])
    monkeypatch.setattr(loop_module, "run_turn_stream", _fake_stream)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        # input() raises StopIteration once the fake iterator is exhausted, which
        # bubbles out of run_chat_loop's own try/except (it only catches
        # EOFError/KeyboardInterrupt) -- fine here, we only need one round to happen
        # first, and the finally block still runs and closes usage_store correctly.
        run_chat_loop(
            ToolRegistry(),
            session_id="s1",
            model="anthropic:claude-sonnet-5",
            usage_store=usage_store,
        )

    summary = UsageStore(tmp_path / "usage.db").summary()
    assert summary["total_calls"] == 1
    row = summary["by_model"][0]
    assert row["provider"] == "anthropic"
    assert row["model"] == "claude-sonnet-5"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50

    out = capsys.readouterr().out
    assert "total" in out  # the "~$X.XXXX total" cost suffix


# ---------------------------------------------------------------------------
# mazu run
# ---------------------------------------------------------------------------


def test_run_logs_usage(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])

    usage_store = UsageStore(tmp_path / "usage.db")
    checkpoint_manager = CheckpointManager(tmp_path)

    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    from mazu.agent.autonomous import run_autonomous

    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
        model="deepseek:deepseek-chat",
        usage_store=usage_store,
    )

    summary = UsageStore(tmp_path / "usage.db").summary()
    assert summary["total_calls"] == 1
    assert summary["by_model"][0]["provider"] == "deepseek"


# ---------------------------------------------------------------------------
# mazu council -- the real thread-safety test
# ---------------------------------------------------------------------------


def test_council_logs_usage_from_parallel_members_without_crashing(tmp_path, monkeypatch):
    usage_store = UsageStore(tmp_path / "usage.db")
    models = [
        "anthropic:claude-sonnet-5",
        "deepseek:deepseek-chat",
        "openai:gpt-5",
        "anthropic:claude-haiku-4-5",
    ]

    def _fake_run_turn(messages, system, tools, model=None):
        # Every member and the lead go through this. Real work happens on a real
        # thread pool (run_council's own ThreadPoolExecutor, not mocked) -- this is
        # what actually exercises the sqlite3 single-thread-connection risk.
        return _end_turn_response(f"answer from {model}")

    with patch("mazu.agent.council.run_turn", side_effect=_fake_run_turn):
        run_council(
            "should we use X or Y?",
            models=models,
            lead_model="anthropic:claude-sonnet-5",
            full_registry=ToolRegistry(),
            usage_store=usage_store,
            session_id="s1",
        )

    summary = UsageStore(tmp_path / "usage.db").summary()
    # 4 members (1 call each, since the fake ends the round immediately with
    # stop_reason="end_turn") + 1 lead call = 5.
    assert summary["total_calls"] == 5
    assert {row["model"] for row in summary["by_model"]} == {
        "claude-sonnet-5",
        "deepseek-chat",
        "gpt-5",
        "claude-haiku-4-5",
    }
