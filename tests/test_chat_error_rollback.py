"""Real bug found via live testing (mazu-web's image-upload feature, which
mirrors mazu/agent/loop.py's turn handling closely): a turn that fails with a
non-retryable MazuAPIError used to leave that turn's user message stuck in
`messages` forever. Since every LLM call resends the full history, every later
turn kept resending the bad message and re-triggering the exact same error,
permanently breaking the chat session. This file verifies _run_until_done's
messages_before_turn rollback actually fixes that in mazu/agent/loop.py.
"""

import pytest

import mazu.agent.loop as loop_module
from mazu.agent.loop import run_chat_loop
from mazu.llm.errors import MazuAPIError
from mazu.llm.types import AgentResponse
from mazu.tools.registry import ToolRegistry


def _end_turn_response(text="done") -> AgentResponse:
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": text}], usage={})


def test_failed_turn_does_not_poison_the_next_turn(monkeypatch):
    registry = ToolRegistry()
    calls_seen: list[list[dict]] = []

    def _fake_stream(messages, system, tools, on_delta, model=None):
        calls_seen.append([dict(m) for m in messages])
        if len(calls_seen) == 1:
            raise MazuAPIError("this model does not support image")
        return _end_turn_response("ok")

    monkeypatch.setattr(loop_module, "run_turn_stream", _fake_stream)
    monkeypatch.setattr(loop_module, "compact_if_needed", lambda messages, model, **k: (messages, False))
    inputs = iter(["first message (will fail)", "second message (should succeed)", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5")

    assert len(calls_seen) == 2
    second_call_messages = calls_seen[1]
    # The failed first turn's user message must be gone by the second call --
    # otherwise it's resent forever and keeps re-triggering the same error.
    assert not any(m.get("content") == "first message (will fail)" for m in second_call_messages)
    assert any(m.get("content") == "second message (should succeed)" for m in second_call_messages)
    # And the second call must be a clean single-turn request, not carrying any
    # leftover assistant/tool residue from the failed attempt.
    assert second_call_messages == [{"role": "user", "content": "second message (should succeed)"}]


def test_error_message_still_printed_on_failed_turn(monkeypatch, capsys):
    registry = ToolRegistry()

    def _always_fails(messages, system, tools, on_delta, model=None):
        raise MazuAPIError("this model does not support image")

    monkeypatch.setattr(loop_module, "run_turn_stream", _always_fails)
    monkeypatch.setattr(loop_module, "compact_if_needed", lambda messages, model, **k: (messages, False))
    inputs = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5")

    assert "this model does not support image" in capsys.readouterr().out
