"""Verifies `mazu chat` (mazu/agent/loop.py) wires the compaction module in
correctly: proactive compact_if_needed() runs every round, a
MazuContextLengthError triggers force_compact() plus exactly one retry, and the
chat-specific extras (ChatStore untouched, memory extraction on compaction,
cost logging, the manual /compact command) all work end to end. The compaction
algorithm's own correctness is covered separately in test_compaction.py --
this file is only about the wiring/control-flow in loop.py, so compaction
itself is stubbed out where the test isn't specifically about it.
"""

import pytest

import mazu.agent.loop as loop_module
from mazu.agent.loop import _handle_compact, run_chat_loop
from mazu.chat.store import ChatStore
from mazu.llm.errors import MazuContextLengthError
from mazu.llm.types import AgentResponse
from mazu.memory.store import MemoryStore
from mazu.tools.registry import ToolRegistry
from mazu.usage.store import UsageStore


def _end_turn_response(text="done") -> AgentResponse:
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": text}], usage={})


def test_proactive_compaction_checked_every_round(monkeypatch):
    registry = ToolRegistry()
    calls = []

    def _fake_compact_if_needed(messages, model, **kwargs):
        calls.append(len(messages))
        return messages, False

    monkeypatch.setattr(loop_module, "compact_if_needed", _fake_compact_if_needed)
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: _end_turn_response())
    inputs = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5")

    assert len(calls) == 1


def test_context_length_error_triggers_force_compact_and_retries_once(monkeypatch):
    registry = ToolRegistry()
    run_calls = {"n": 0}

    def _fake_stream(messages, system, tools, on_delta, model=None):
        run_calls["n"] += 1
        if run_calls["n"] == 1:
            raise MazuContextLengthError("too long")
        return _end_turn_response("recovered")

    force_compact_calls = {"n": 0}

    def _fake_force_compact(messages, model, **kwargs):
        force_compact_calls["n"] += 1
        return messages[-1:]

    monkeypatch.setattr(loop_module, "run_turn_stream", _fake_stream)
    monkeypatch.setattr(loop_module, "force_compact", _fake_force_compact)
    monkeypatch.setattr(loop_module, "compact_if_needed", lambda messages, model, **k: (messages, False))
    inputs = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5")

    assert run_calls["n"] == 2
    assert force_compact_calls["n"] == 1


def test_context_length_error_on_retry_falls_through_to_generic_error(monkeypatch, capsys):
    registry = ToolRegistry()

    def _always_raises(messages, system, tools, on_delta, model=None):
        raise MazuContextLengthError("still too long")

    monkeypatch.setattr(loop_module, "run_turn_stream", _always_raises)
    monkeypatch.setattr(loop_module, "force_compact", lambda messages, model, **k: messages)
    monkeypatch.setattr(loop_module, "compact_if_needed", lambda messages, model, **k: (messages, False))
    inputs = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    # Must not raise MazuContextLengthError/MazuAPIError past this point -- falls
    # through to the same "[error] ... Returning to the prompt" handling text, then
    # the loop continues to its next input() call, which is what raises StopIteration.
    with pytest.raises(StopIteration):
        run_chat_loop(registry, session_id="s1", model="anthropic:claude-sonnet-5")

    assert "[error]" in capsys.readouterr().out


def test_chat_store_transcript_untouched_by_compaction(tmp_path, monkeypatch):
    registry = ToolRegistry()
    chat_store = ChatStore(tmp_path / "chat_history.db")

    responses = iter([_end_turn_response("reply one"), _end_turn_response("reply two")])
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: next(responses))
    # Force the REAL compact_if_needed/compact_messages algorithm to run on every
    # round (needs_compaction forced True instead of relying on the real 60k-token
    # threshold), with only the LLM summarization call itself stubbed out --
    # monkeypatched on compaction.py's own namespace, since that's where
    # compact_if_needed's internal lambda resolves these names from, not loop.py's.
    monkeypatch.setattr("mazu.agent.compaction.needs_compaction", lambda messages, trigger_tokens: True)
    monkeypatch.setattr(
        "mazu.agent.compaction.summarize_for_compaction", lambda mm, model, **k: "SUMMARY"
    )
    inputs = iter(["first message", "second message", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry, session_id="s1", model="anthropic:claude-sonnet-5", chat_store=chat_store,
        )

    # Whatever compaction did to the in-memory list, ChatStore has every raw
    # message: 2 user turns + 2 assistant replies = 4, never trimmed.
    persisted = ChatStore(tmp_path / "chat_history.db").get_messages("s1")
    assert len(persisted) == 4


def test_compaction_extracts_memories_from_the_compacted_chunk(tmp_path, monkeypatch):
    registry = ToolRegistry()
    memory_store = MemoryStore(tmp_path / "memory.db")

    monkeypatch.setattr(
        "mazu.agent.session.extract_memories",
        lambda messages, model=None: (
            [{"category": "decision", "title": "Use PostgreSQL", "body": "for concurrency"}],
            {},
        ),
    )

    calls = {"n": 0}

    def _fake_compact_if_needed(messages, model, on_compacted=None, **kwargs):
        calls["n"] += 1
        if on_compacted is not None:
            on_compacted(messages)
        return messages, True

    monkeypatch.setattr(loop_module, "compact_if_needed", _fake_compact_if_needed)
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: _end_turn_response())
    inputs = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry, session_id="s1", model="anthropic:claude-sonnet-5", memory_store=None,
        )
    assert calls["n"] >= 1  # sanity: the fake actually ran

    # Re-run with a real memory_store passed through, to prove the wiring reaches it.
    memory_store2 = MemoryStore(tmp_path / "memory2.db")
    inputs2 = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs2))
    with pytest.raises(StopIteration):
        run_chat_loop(
            registry, session_id="s2", model="anthropic:claude-sonnet-5", memory_store=memory_store2,
        )
    # finalize_session (run_chat_loop's finally block) already closed memory_store2 --
    # reopen the same db file, matching test_dedup_regression.py's pattern.
    reopened = MemoryStore(memory_store2.db_path)
    rows = reopened.search()
    reopened.close()
    assert len(rows) == 1
    assert rows[0]["title"] == "Use PostgreSQL"


def test_compaction_logs_real_cost_to_usage_store(tmp_path, monkeypatch):
    registry = ToolRegistry()
    usage_store = UsageStore(tmp_path / "usage.db")

    def _fake_compact_if_needed(messages, model, usage_store=None, session_id=None, on_compacted=None, **kwargs):
        if usage_store is not None:
            usage_store.log("compaction", session_id, "anthropic", "claude-haiku-4-5", 1000, 200, 0.001)
        return messages, True

    monkeypatch.setattr(loop_module, "compact_if_needed", _fake_compact_if_needed)
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: _end_turn_response())
    inputs = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry, session_id="s1", model="anthropic:claude-sonnet-5", usage_store=usage_store,
        )

    summary = UsageStore(tmp_path / "usage.db").summary(command="compaction")
    assert summary["total_calls"] == 1


def test_max_cost_counts_chat_compaction_and_final_memory_extraction(tmp_path, monkeypatch, capsys):
    registry = ToolRegistry()
    usage_store = UsageStore(tmp_path / "usage.db")
    memory_store = MemoryStore(tmp_path / "memory.db")

    def _compact(messages, model, usage_store=None, session_id=None, **kwargs):
        usage_store.log("compaction", session_id, "deepseek", "deepseek-chat", 1, 1, 0.0002)
        return messages, False

    def _finalize(store, session_id, messages, usage_store=None, **kwargs):
        usage_store.log("memory_extraction", session_id, "deepseek", "deepseek-chat", 1, 1, 0.0006)
        store.close()
        return 0

    monkeypatch.setattr(loop_module, "compact_if_needed", _compact)
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: AgentResponse(
        stop_reason="end_turn", content=[{"type": "text", "text": "done"}],
        usage={"input_tokens": 1_000, "output_tokens": 0},
    ))
    monkeypatch.setattr(loop_module, "finalize_session", _finalize)
    inputs = iter(["hello"])

    def _input_with_eof(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", _input_with_eof)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    run_chat_loop(
        registry, session_id="s1", model="deepseek:deepseek-chat", usage_store=usage_store,
        memory_store=memory_store, max_cost=0.001,
    )

    summary = UsageStore(tmp_path / "usage.db").summary(session_id="s1")
    assert summary["total_calls"] == 3
    assert summary["total_cost"] == pytest.approx(0.00107)
    output = capsys.readouterr().out
    assert "reached --max-cost" in output
    assert "final session total" in output


def test_handle_compact_shrinks_a_long_conversation(monkeypatch, capsys):
    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"task {i}"})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"response {i}"}]})

    monkeypatch.setattr(loop_module, "summarize_for_compaction", lambda mm, model, **k: "SUMMARY")

    before = len(messages)
    _handle_compact(messages, model=None, memory_store=None, usage_store=None, session_id="s1")

    assert len(messages) < before
    out = capsys.readouterr().out
    assert "[compact]" in out
    assert "still saved" in out


def test_handle_compact_noop_on_a_short_conversation(capsys):
    messages = [{"role": "user", "content": "hi"}]
    _handle_compact(messages, model=None, memory_store=None, usage_store=None, session_id="s1")

    assert len(messages) == 1
    assert "nothing to compact yet" in capsys.readouterr().out
