"""Verifies mazu chat / mazu run / mazu council all correctly log tool calls to
ActionLogStore, including the branches that never reach a tool's real handler
(unknown tool, denylisted shell command, user-declined destructive tool) -- and, like
UsageStore, that mazu council's parallel worker threads never touch the
ActionLogStore's sqlite3 connection directly (writes happen back in the main thread).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import mazu.agent.autonomous as autonomous_module
import mazu.agent.loop as loop_module
from mazu.action_log.store import ActionLogStore
from mazu.agent.council import run_council
from mazu.agent.loop import run_chat_loop
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.types import AgentResponse
from mazu.tools.base import Tool, ToolResult
from mazu.tools.registry import ToolRegistry


def _tool_use_response(tool_name: str, tool_input: dict, tool_use_id: str = "t1") -> AgentResponse:
    return AgentResponse(
        stop_reason="tool_use",
        content=[{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": tool_input}],
        usage={"input_tokens": 100, "output_tokens": 50},
    )


def _end_turn_response(text: str = "done") -> AgentResponse:
    return AgentResponse(
        stop_reason="end_turn",
        content=[{"type": "text", "text": text}],
        usage={"input_tokens": 100, "output_tokens": 50},
    )


def _read_file_tool(content: str = "file contents", is_error: bool = False) -> Tool:
    return Tool(
        name="read_file",
        description="read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=lambda inp: ToolResult(content, is_error=is_error),
        destructive=False,
    )


def _write_file_tool() -> Tool:
    return Tool(
        name="write_file",
        description="write a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=lambda inp: ToolResult(f"Wrote to {inp['path']}"),
        destructive=True,
    )


# ---------------------------------------------------------------------------
# mazu chat
# ---------------------------------------------------------------------------


def test_chat_logs_successful_tool_call(tmp_path, monkeypatch):
    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    registry = ToolRegistry()
    registry.register(_read_file_tool())

    responses = iter([_tool_use_response("read_file", {"path": "a.py"}), _end_turn_response()])

    def _fake_stream(messages, system, tools, on_delta, model=None):
        return next(responses)

    inputs = iter(["read a.py", ""])
    monkeypatch.setattr(loop_module, "run_turn_stream", _fake_stream)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry,
            session_id="s1",
            model="anthropic:claude-sonnet-5",
            action_log_store=action_log_store,
        )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert len(actions) == 1
    assert actions[0]["tool_name"] == "read_file"
    assert actions[0]["outcome"] == "ok"
    assert actions[0]["command"] == "chat"


def test_chat_logs_unknown_tool(tmp_path, monkeypatch):
    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    registry = ToolRegistry()

    responses = iter([_tool_use_response("does_not_exist", {}), _end_turn_response()])
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    inputs = iter(["do something", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry, session_id="s1", model="anthropic:claude-sonnet-5", action_log_store=action_log_store
        )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert actions[0]["outcome"] == "unknown_tool"


def test_chat_logs_denylisted_shell_command(tmp_path, monkeypatch):
    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="run_shell",
            description="run a shell command",
            input_schema={"type": "object"},
            handler=lambda inp: ToolResult("should never be called"),
            destructive=True,
        )
    )

    responses = iter(
        [_tool_use_response("run_shell", {"command": "sudo rm -rf /"}), _end_turn_response()]
    )
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    inputs = iter(["delete everything", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry, session_id="s1", model="anthropic:claude-sonnet-5", action_log_store=action_log_store
        )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert actions[0]["outcome"] == "blocked"


def test_chat_logs_declined_destructive_tool(tmp_path, monkeypatch):
    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    registry = ToolRegistry()
    registry.register(_write_file_tool())

    responses = iter([_tool_use_response("write_file", {"path": "a.py"}), _end_turn_response()])
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    monkeypatch.setattr(loop_module, "safe_confirm", lambda prompt: False)
    inputs = iter(["write a.py", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry, session_id="s1", model="anthropic:claude-sonnet-5", action_log_store=action_log_store
        )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert actions[0]["outcome"] == "declined"


def test_chat_action_log_store_none_does_not_crash(monkeypatch):
    registry = ToolRegistry()
    registry.register(_read_file_tool())

    responses = iter([_tool_use_response("read_file", {"path": "a.py"}), _end_turn_response()])
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    inputs = iter(["read a.py", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5")


# ---------------------------------------------------------------------------
# mazu run
# ---------------------------------------------------------------------------


def test_run_logs_tool_calls_including_changed_file(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])

    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    checkpoint_manager = CheckpointManager(tmp_path)
    registry = ToolRegistry()
    registry.register(_write_file_tool())

    responses = iter(
        [_tool_use_response("write_file", {"path": "a.py", "content": "x"}), _end_turn_response()]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))
    monkeypatch.setattr(autonomous_module, "safe_confirm", lambda prompt: True)

    from mazu.agent.autonomous import run_autonomous

    run_autonomous(
        registry=registry,
        task="write a.py",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=2,
        model="deepseek:deepseek-chat",
        action_log_store=action_log_store,
    )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert len(actions) == 1
    assert actions[0]["tool_name"] == "write_file"
    assert actions[0]["changed_file"] == "a.py"
    assert actions[0]["command"] == "run"


def test_run_logs_tool_error(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])

    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    checkpoint_manager = CheckpointManager(tmp_path)
    registry = ToolRegistry()
    registry.register(_read_file_tool(content="boom", is_error=True))

    responses = iter([_tool_use_response("read_file", {"path": "missing.py"}), _end_turn_response()])
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    from mazu.agent.autonomous import run_autonomous

    run_autonomous(
        registry=registry,
        task="read missing.py",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=2,
        model="deepseek:deepseek-chat",
        action_log_store=action_log_store,
    )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert actions[0]["outcome"] == "error"


# ---------------------------------------------------------------------------
# mazu council -- the real thread-safety test
# ---------------------------------------------------------------------------


def test_council_logs_tool_calls_from_parallel_members_without_crashing(tmp_path, monkeypatch):
    action_log_store = ActionLogStore(tmp_path / "action_log.db")
    models = [
        "anthropic:claude-sonnet-5",
        "deepseek:deepseek-chat",
        "openai:gpt-5",
        "anthropic:claude-haiku-4-5",
    ]
    registry = ToolRegistry()
    registry.register(_read_file_tool())

    call_counters: dict[str, int] = {}

    def _fake_run_turn(messages, system, tools, model=None):
        # First call per member triggers a tool call, second ends the round --
        # real work happens on a real ThreadPoolExecutor (run_council's own, not
        # mocked), which is what actually exercises the sqlite3 thread-safety risk.
        count = call_counters.get(model, 0)
        call_counters[model] = count + 1
        if count == 0:
            return _tool_use_response("read_file", {"path": "a.py"})
        return _end_turn_response(f"answer from {model}")

    with patch("mazu.agent.council.run_turn", side_effect=_fake_run_turn):
        run_council(
            "should we use X or Y?",
            models=models,
            lead_model="anthropic:claude-sonnet-5",
            full_registry=registry,
            session_id="s1",
            action_log_store=action_log_store,
        )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    # One read_file tool call per member, all logged from the main thread without
    # crashing or losing rows despite running in 4 parallel worker threads.
    assert len(actions) == 4
    assert all(a["tool_name"] == "read_file" for a in actions)
    assert all(a["command"] == "council" for a in actions)


def test_council_action_log_store_none_does_not_crash(monkeypatch):
    registry = ToolRegistry()
    registry.register(_read_file_tool())

    def _fake_run_turn(messages, system, tools, model=None):
        return _end_turn_response(f"answer from {model}")

    with patch("mazu.agent.council.run_turn", side_effect=_fake_run_turn):
        run_council(
            "question",
            models=["anthropic:claude-sonnet-5"],
            lead_model="anthropic:claude-sonnet-5",
            full_registry=registry,
            session_id="s1",
        )
