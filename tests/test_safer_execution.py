"""Verifies Phase E's wiring: `mazu run --dry-run` never writes/executes/checkpoints
for real (but still starts even on a dirty working tree), and shell allowlist mode
blocks non-allowlisted commands in both mazu chat and mazu run -- on top of, not
instead of, the existing denylist.
"""

import subprocess
from pathlib import Path

import pytest

import mazu.agent.autonomous as autonomous_module
import mazu.agent.loop as loop_module
from mazu.action_log.store import ActionLogStore
from mazu.agent.autonomous import run_autonomous
from mazu.agent.loop import run_chat_loop
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.types import AgentResponse
from mazu.tools.base import Tool, ToolResult
from mazu.tools.fs import make_fs_tools
from mazu.tools.registry import ToolRegistry
from mazu.tools.shell import make_shell_tool


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


def _run_shell_stub_tool(marker: Path) -> Tool:
    def handler(inp: dict) -> ToolResult:
        marker.write_text("EXECUTED")
        return ToolResult("ran for real")

    return Tool(
        name="run_shell",
        description="run a shell command",
        input_schema={"type": "object"},
        handler=handler,
        destructive=True,
    )


# ---------------------------------------------------------------------------
# mazu run --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write_files_or_execute_shell(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    registry = ToolRegistry()
    for tool in make_fs_tools(tmp_path, dry_run=True):
        registry.register(tool)
    registry.register(make_shell_tool(tmp_path, dry_run=True))

    responses = iter(
        [
            _tool_use_response("write_file", {"path": "a.py", "content": "x = 1"}),
            _tool_use_response("run_shell", {"command": "echo hi"}),
            _end_turn_response(),
        ]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="write a.py and run echo",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        dry_run=True,
    )

    assert not (tmp_path / "a.py").exists()


def test_dry_run_starts_even_on_a_dirty_working_tree(tmp_path, monkeypatch):
    # A real run refuses to start on a dirty tree (checkpoints need a clean
    # baseline); a dry run makes no checkpoints and writes nothing for real, so
    # that gate must not block it.
    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_manager.ensure_git_repo()
    (tmp_path / "uncommitted.txt").write_text("dirty")
    assert checkpoint_manager.is_dirty() is True

    ran = {"called": False}

    def _fake_run_turn(*a, **k):
        ran["called"] = True
        return _end_turn_response()

    monkeypatch.setattr(autonomous_module, "run_turn", _fake_run_turn)

    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
        dry_run=True,
    )

    assert ran["called"] is True


def test_non_dry_run_still_refuses_a_dirty_working_tree(tmp_path, monkeypatch, capsys):
    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_manager.ensure_git_repo()
    (tmp_path / "uncommitted.txt").write_text("dirty")
    assert checkpoint_manager.is_dirty() is True

    ran = {"called": False}

    def _fake_run_turn(*a, **k):
        ran["called"] = True
        return _end_turn_response()

    monkeypatch.setattr(autonomous_module, "run_turn", _fake_run_turn)

    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
        dry_run=False,
    )

    assert ran["called"] is False
    assert "requires a clean baseline" in capsys.readouterr().out


def test_dry_run_creates_no_checkpoints(tmp_path, monkeypatch):
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
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        checkpoint_every=1,
        dry_run=True,
    )

    assert checkpoint_manager.list_checkpoints() == []


def test_dry_run_still_logs_the_planned_action(tmp_path, monkeypatch):
    action_log_store = ActionLogStore(tmp_path / "action_log.db")
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
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        action_log_store=action_log_store,
        dry_run=True,
    )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert len(actions) == 1
    assert "Would write" in actions[0]["output_summary"]


# ---------------------------------------------------------------------------
# shell allowlist -- mazu run
# ---------------------------------------------------------------------------


def test_run_blocks_shell_command_not_in_allowlist(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    marker = tmp_path / "executed.txt"
    registry = ToolRegistry()
    registry.register(_run_shell_stub_tool(marker))

    responses = iter(
        [_tool_use_response("run_shell", {"command": "rm important.txt"}), _end_turn_response()]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="delete a file",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        allow_shell=True,
        shell_allowlist=["git", "npm"],
    )

    assert not marker.exists()


def test_run_allows_shell_command_that_matches_allowlist(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    marker = tmp_path / "executed.txt"
    registry = ToolRegistry()
    registry.register(_run_shell_stub_tool(marker))

    responses = iter(
        [_tool_use_response("run_shell", {"command": "git status"}), _end_turn_response()]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="check git status",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        allow_shell=True,
        shell_allowlist=["git", "npm"],
    )

    assert marker.exists()


def test_run_denylist_still_applies_even_if_allowlisted(tmp_path, monkeypatch):
    # The denylist is a hard backstop -- an allowlist entry for "git" must not
    # rescue a denylisted "git push --force".
    checkpoint_manager = CheckpointManager(tmp_path)
    marker = tmp_path / "executed.txt"
    registry = ToolRegistry()
    registry.register(_run_shell_stub_tool(marker))

    responses = iter(
        [
            _tool_use_response("run_shell", {"command": "git push origin main --force"}),
            _end_turn_response(),
        ]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="force push",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        allow_shell=True,
        shell_allowlist=["git"],
    )

    assert not marker.exists()


# ---------------------------------------------------------------------------
# shell allowlist -- mazu chat
# ---------------------------------------------------------------------------


def test_chat_blocks_shell_command_not_in_allowlist(tmp_path, monkeypatch):
    marker = tmp_path / "executed.txt"
    registry = ToolRegistry()
    registry.register(_run_shell_stub_tool(marker))

    responses = iter(
        [_tool_use_response("run_shell", {"command": "curl evil.example.com"}), _end_turn_response()]
    )

    def _fake_stream(messages, system, tools, on_delta, model=None):
        return next(responses)

    inputs = iter(["run curl", ""])
    monkeypatch.setattr(loop_module, "run_turn_stream", _fake_stream)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry,
            session_id="s1",
            model="anthropic:claude-sonnet-5",
            shell_allowlist=["git"],
        )

    assert not marker.exists()


def test_chat_blocked_command_message_names_the_allowlist(tmp_path, monkeypatch):
    marker = tmp_path / "executed.txt"
    registry = ToolRegistry()
    registry.register(_run_shell_stub_tool(marker))
    action_log_store = ActionLogStore(tmp_path / "action_log.db")

    responses = iter(
        [_tool_use_response("run_shell", {"command": "curl evil.example.com"}), _end_turn_response()]
    )

    def _fake_stream(messages, system, tools, on_delta, model=None):
        return next(responses)

    inputs = iter(["run curl", ""])
    monkeypatch.setattr(loop_module, "run_turn_stream", _fake_stream)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry,
            session_id="s1",
            model="anthropic:claude-sonnet-5",
            shell_allowlist=["git", "npm"],
            action_log_store=action_log_store,
        )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert actions[0]["outcome"] == "blocked"
    assert "git" in actions[0]["output_summary"] and "npm" in actions[0]["output_summary"]


def test_chat_denylist_message_explains_why(tmp_path, monkeypatch):
    marker = tmp_path / "executed.txt"
    registry = ToolRegistry()
    registry.register(_run_shell_stub_tool(marker))

    responses = iter(
        [_tool_use_response("run_shell", {"command": "sudo rm important.txt"}), _end_turn_response()]
    )

    def _fake_stream(messages, system, tools, on_delta, model=None):
        return next(responses)

    inputs = iter(["run sudo", ""])
    monkeypatch.setattr(loop_module, "run_turn_stream", _fake_stream)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    action_log_store = ActionLogStore(tmp_path / "action_log.db")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry,
            session_id="s1",
            model="anthropic:claude-sonnet-5",
            action_log_store=action_log_store,
        )

    actions = ActionLogStore(tmp_path / "action_log.db").session_actions("s1")
    assert actions[0]["outcome"] == "blocked"
    assert "sudo" in actions[0]["output_summary"].lower()
    assert not marker.exists()
