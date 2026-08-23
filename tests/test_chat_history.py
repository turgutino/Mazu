"""Verifies chat-history persistence end to end: run_chat_loop appends every
message to ChatStore as it happens (not just on manual /checkpoint), and `mazu
chat --list`/`--resume` can list and replay a saved session. Companion to
tests/test_chat_store.py (which covers ChatStore's own storage logic in
isolation) and tests/test_action_log_wiring.py (same stubbing pattern, applied
to ActionLogStore instead).
"""

import subprocess

import pytest
from click.testing import CliRunner

import mazu.agent.loop as loop_module
import mazu.cli as cli_module
from mazu.chat.store import ChatStore
from mazu.cli import main
from mazu.llm.types import AgentResponse
from mazu.memory.store import MemoryStore
from mazu.tools.registry import ToolRegistry


def _end_turn_response(text: str = "done") -> AgentResponse:
    return AgentResponse(
        stop_reason="end_turn",
        content=[{"type": "text", "text": text}],
        usage={"input_tokens": 100, "output_tokens": 50},
    )


def _tool_use_response(tool_name: str, tool_input: dict, tool_use_id: str = "t1") -> AgentResponse:
    return AgentResponse(
        stop_reason="tool_use",
        content=[{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": tool_input}],
        usage={"input_tokens": 100, "output_tokens": 50},
    )


# ---------------------------------------------------------------------------
# run_chat_loop -> ChatStore wiring
# ---------------------------------------------------------------------------


def test_run_chat_loop_persists_user_and_assistant_messages_in_order(tmp_path, monkeypatch):
    chat_store = ChatStore(tmp_path / "chat_history.db")
    registry = ToolRegistry()

    responses = iter([_end_turn_response("hi there")])
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    inputs = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    from mazu.agent.loop import run_chat_loop

    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5", chat_store=chat_store)

    messages = ChatStore(tmp_path / "chat_history.db").get_messages("s1")
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
    ]


def test_run_chat_loop_persists_tool_result_round_before_the_final_answer(tmp_path, monkeypatch):
    from mazu.tools.base import Tool, ToolResult

    chat_store = ChatStore(tmp_path / "chat_history.db")
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=lambda inp: ToolResult("file contents"),
            destructive=False,
        )
    )

    responses = iter([_tool_use_response("read_file", {"path": "a.py"}), _end_turn_response()])
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    inputs = iter(["read a.py", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    from mazu.agent.loop import run_chat_loop

    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5", chat_store=chat_store)

    messages = ChatStore(tmp_path / "chat_history.db").get_messages("s1")
    roles_and_shapes = [(m["role"], type(m["content"]).__name__) for m in messages]
    # user text, assistant tool_use, user tool_result, assistant final text -- every
    # step of the round persisted, not just the turn's final answer.
    assert roles_and_shapes == [
        ("user", "str"),
        ("assistant", "list"),
        ("user", "list"),
        ("assistant", "list"),
    ]


def test_run_chat_loop_without_a_chat_store_does_not_crash(monkeypatch):
    registry = ToolRegistry()
    responses = iter([_end_turn_response()])
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    inputs = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    from mazu.agent.loop import run_chat_loop

    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5")  # no chat_store kwarg


def test_run_chat_loop_seeds_messages_from_resumed_messages(monkeypatch):
    registry = ToolRegistry()
    captured_messages = {}

    def _fake_stream(messages, system, tools, on_delta, model=None):
        captured_messages["seen"] = list(messages)
        return _end_turn_response()

    monkeypatch.setattr(loop_module, "run_turn_stream", _fake_stream)
    inputs = iter(["continue please", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    prior = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": [{"type": "text", "text": "earlier answer"}]},
    ]

    from mazu.agent.loop import run_chat_loop

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry, session_id="s1", model="anthropic:claude-sonnet-5", resumed_messages=prior
        )

    # The new user turn is appended AFTER the resumed history, not replacing it.
    assert captured_messages["seen"][:2] == prior
    assert captured_messages["seen"][2] == {"role": "user", "content": "continue please"}


def test_resume_does_not_reinsert_into_memory_sessions_table(tmp_path, monkeypatch):
    # Real bug found via live testing (2026-08-23): resuming a chat crashed with
    # sqlite3.IntegrityError ("UNIQUE constraint failed: sessions.id") because
    # run_chat_loop unconditionally called memory_store.start_session(session_id)
    # on the first new message -- but a resumed session reuses the SAME session_id,
    # whose `sessions` row already exists from the original process. Identical fix
    # (and identical regression-test shape) to autonomous.py's run_autonomous(),
    # see test_run_report_resume.py::test_resume_does_not_reinsert_into_memory_sessions_table.
    registry = ToolRegistry()
    memory_store = MemoryStore(tmp_path / "memory.db")
    memory_store.start_session("s1")  # simulates the original process having already started it

    responses = iter([_end_turn_response("continuing")])
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    inputs = iter(["continue please", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    from mazu.agent.loop import run_chat_loop

    prior = [{"role": "user", "content": "original message"}]

    # Must not raise sqlite3.IntegrityError from a duplicate PRIMARY KEY insert.
    with pytest.raises(StopIteration):
        run_chat_loop(
            registry,
            session_id="s1",
            model="anthropic:claude-sonnet-5",
            memory_store=memory_store,
            resumed_messages=prior,
        )


# ---------------------------------------------------------------------------
# `mazu chat --list` / `mazu chat --resume`
# ---------------------------------------------------------------------------


def _init_git_repo(root, monkeypatch):
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("USERPROFILE", str(root))
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)


def test_chat_list_reports_no_sessions_on_a_fresh_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--list"])
    assert result.exit_code == 0
    assert "No chat sessions yet" in result.output


def test_chat_list_shows_a_previously_saved_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chat_store = ChatStore(tmp_path / ".mazu" / "chat_history.db")
    chat_store.append_message("s1", "user", "help me refactor this function")
    chat_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--list"])
    assert result.exit_code == 0
    assert "s1" in result.output
    assert "help me refactor this function" in result.output


def test_chat_list_json_emits_a_json_array(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chat_store = ChatStore(tmp_path / ".mazu" / "chat_history.db")
    chat_store.append_message("s1", "user", "hello")
    chat_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--list", "--json"])
    assert result.exit_code == 0
    import json

    payload = json.loads(result.output)
    assert payload["data"][0]["session_id"] == "s1"
    assert payload["data"][0]["message_count"] == 1


def test_chat_delete_removes_a_saved_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chat_store = ChatStore(tmp_path / ".mazu" / "chat_history.db")
    chat_store.append_message("s1", "user", "hello")
    chat_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--delete", "s1"])
    assert result.exit_code == 0
    assert "Deleted chat session s1" in result.output

    chat_store = ChatStore(tmp_path / ".mazu" / "chat_history.db")
    assert chat_store.get_session("s1") is None
    chat_store.close()


def test_chat_delete_of_unknown_session_is_a_clean_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--delete", "never-existed"])
    assert result.exit_code == 0
    assert "No chat session found with id never-existed" in result.output


def test_chat_resume_with_no_saved_messages_errors_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--resume", "does-not-exist", "--model", "deepseek:deepseek-chat"])
    assert result.exit_code == 0
    assert "No saved messages found" in result.output


def test_chat_resume_passes_saved_history_into_run_chat_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    chat_store = ChatStore(tmp_path / ".mazu" / "chat_history.db")
    chat_store.append_message("s1", "user", "earlier question")
    chat_store.append_message("s1", "assistant", [{"type": "text", "text": "earlier answer"}])
    chat_store.close()

    captured = {}

    def _fake_run_chat_loop(registry, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "run_chat_loop", _fake_run_chat_loop)

    runner = CliRunner()
    result = runner.invoke(
        main, ["chat", "--resume", "s1", "--model", "deepseek:deepseek-chat"]
    )

    assert result.exit_code == 0, result.output
    assert captured["session_id"] == "s1"  # continues the SAME session, doesn't mint a new one
    assert captured["resumed_messages"] == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": [{"type": "text", "text": "earlier answer"}]},
    ]
