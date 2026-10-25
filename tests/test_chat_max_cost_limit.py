"""Behavioural guarantees of `mazu chat --max-cost`, beyond the totals checked in
test_chat_compaction.py: the limit must actually STOP further model calls, must not
leave the saved conversation in a state the API would reject (a tool_use with no
tool_result), and must be enforced even when earlier spend already exceeds it.
"""

import mazu.agent.loop as loop_module
from mazu.agent.loop import run_chat_loop
from mazu.llm.types import AgentResponse
from mazu.tools.registry import ToolRegistry
from mazu.usage.store import UsageStore


def _feed(monkeypatch, lines):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")


def _tool_use_response(tokens=1_000_000):
    return AgentResponse(
        stop_reason="tool_use",
        content=[{"type": "tool_use", "id": "t1", "name": "missing_tool", "input": {}}],
        usage={"input_tokens": tokens, "output_tokens": 0},
    )


def test_limit_stops_the_loop_and_closes_open_tool_use(tmp_path, monkeypatch):
    calls = []

    def fake_stream(*args, **kwargs):
        calls.append(1)
        return _tool_use_response()  # deepseek-chat input $0.27/M -> ~$0.27 per call

    monkeypatch.setattr(loop_module, "run_turn_stream", fake_stream)
    monkeypatch.setattr(loop_module, "compact_if_needed", lambda m, model, **k: (m, False))
    monkeypatch.setattr(loop_module, "finalize_session", lambda *a, **k: 0)
    _feed(monkeypatch, ["do something", "and another thing"])

    usage = UsageStore(tmp_path / "usage.db")
    run_chat_loop(ToolRegistry(), session_id="s1", model="deepseek:deepseek-chat",
                  usage_store=usage, max_cost=0.10)

    # exactly one model call: the limit was crossed by it, so no second call and the
    # second user input was never even sent
    assert len(calls) == 1


def test_open_tool_use_gets_error_result_in_history(tmp_path, monkeypatch):
    captured = {}

    def fake_stream(messages, *args, **kwargs):
        captured["messages"] = messages
        return _tool_use_response()

    monkeypatch.setattr(loop_module, "run_turn_stream", fake_stream)
    monkeypatch.setattr(loop_module, "compact_if_needed", lambda m, model, **k: (m, False))
    monkeypatch.setattr(loop_module, "finalize_session", lambda *a, **k: 0)
    _feed(monkeypatch, ["do something"])

    usage = UsageStore(tmp_path / "usage.db")
    run_chat_loop(ToolRegistry(), session_id="s1", model="deepseek:deepseek-chat",
                  usage_store=usage, max_cost=0.10)

    messages = captured["messages"]
    assert messages[-1]["role"] == "user"
    result = messages[-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert result["is_error"] is True and "--max-cost" in result["content"]


def test_prior_spend_over_limit_blocks_every_model_call(tmp_path, monkeypatch, capsys):
    usage = UsageStore(tmp_path / "usage.db")
    usage.log("chat", "s1", "deepseek", "deepseek-chat", 1, 1, 0.50)  # already spent
    calls = []
    monkeypatch.setattr(loop_module, "run_turn_stream", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(loop_module, "compact_if_needed", lambda m, model, **k: (m, False))
    monkeypatch.setattr(loop_module, "finalize_session", lambda *a, **k: 0)
    _feed(monkeypatch, ["hello"])

    run_chat_loop(ToolRegistry(), session_id="s1", model="deepseek:deepseek-chat",
                  usage_store=usage, max_cost=0.10)

    assert calls == []
    assert "already reached --max-cost" in capsys.readouterr().out


def test_no_limit_means_no_behaviour_change(tmp_path, monkeypatch):
    calls = []

    def fake_stream(*a, **k):
        calls.append(1)
        return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "ok"}],
                             usage={"input_tokens": 5_000_000, "output_tokens": 0})

    monkeypatch.setattr(loop_module, "run_turn_stream", fake_stream)
    monkeypatch.setattr(loop_module, "compact_if_needed", lambda m, model, **k: (m, False))
    monkeypatch.setattr(loop_module, "finalize_session", lambda *a, **k: 0)
    _feed(monkeypatch, ["one", "two"])

    run_chat_loop(ToolRegistry(), session_id="s1", model="deepseek:deepseek-chat",
                  usage_store=UsageStore(tmp_path / "usage.db"))

    assert len(calls) == 2  # an expensive first reply does not stop a chat with no limit
