"""Tests for Addendum 6's multi-breakpoint Anthropic prompt caching: cache_control
must land on the last tool definition and the last message (whether its content is a
plain string or a list of content blocks), and the caller's original `tools`/
`messages` objects must never be mutated -- they're the same objects reused verbatim
by checkpoint snapshots, context compaction, and the OpenAI-compatible converter.
"""

import copy
from unittest.mock import MagicMock

from mazu.llm.providers.anthropic_provider import (
    AnthropicProvider,
    _with_cache_control,
    _with_tool_cache_control,
)


def _make_response(stop_reason="end_turn", content=None, usage=None):
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = []
    for block in content or [{"type": "text", "text": "ok"}]:
        b = MagicMock()
        b.model_dump.return_value = block
        response.content.append(b)
    response.usage.model_dump.return_value = usage or {"input_tokens": 1, "output_tokens": 1}
    return response


# ---------------------------------------------------------------------------
# _with_cache_control / _with_tool_cache_control: pure helper correctness
# ---------------------------------------------------------------------------


def test_with_cache_control_wraps_string_content():
    message = {"role": "user", "content": "hello"}
    result = _with_cache_control(message)

    assert result["content"] == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    ]
    assert message["content"] == "hello"  # original untouched


def test_with_cache_control_adds_to_last_block_of_list_content():
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "a"}, {"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
    }
    original = copy.deepcopy(message)

    result = _with_cache_control(message)

    assert result["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in result["content"][0]  # only the LAST block
    assert message == original  # original message and its nested blocks untouched


def test_with_tool_cache_control_marks_only_last_tool():
    tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    original = copy.deepcopy(tools)

    result = _with_tool_cache_control(tools)

    assert "cache_control" not in result[0]
    assert "cache_control" not in result[1]
    assert result[2]["cache_control"] == {"type": "ephemeral"}
    assert tools == original  # original list and dicts untouched


def test_with_tool_cache_control_empty_list_is_a_noop():
    assert _with_tool_cache_control([]) == []


# ---------------------------------------------------------------------------
# run_turn: caching wired into the real request
# ---------------------------------------------------------------------------


def test_run_turn_sends_cache_control_on_system_tools_and_last_message(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _make_response()
    provider._client = fake_client

    messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    tools = [{"name": "read_file"}, {"name": "write_file"}]
    original_messages = copy.deepcopy(messages)
    original_tools = copy.deepcopy(tools)

    provider.run_turn(messages, "system prompt", tools, "claude-sonnet-5")

    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["tools"][0].get("cache_control") is None
    # Last message's last content block gets the breakpoint.
    last_sent_message = call_kwargs["messages"][-1]
    assert last_sent_message["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Earlier messages are unaffected.
    assert call_kwargs["messages"][0] == {"role": "user", "content": "first turn"}

    # The caller's original objects must be byte-for-byte unmutated.
    assert messages == original_messages
    assert tools == original_tools


def test_run_turn_handles_empty_messages_and_tools(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _make_response()
    provider._client = fake_client

    # Must not raise on an empty messages/tools list (e.g. a fresh council lead call).
    provider.run_turn([], "system prompt", [], "claude-sonnet-5")

    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["messages"] == []
    assert call_kwargs["tools"] == []


# ---------------------------------------------------------------------------
# run_turn_stream: same caching guarantees, via the streaming call path
# ---------------------------------------------------------------------------


class _FakeStreamCtx:
    def __init__(self, final):
        self._final = final

    @property
    def text_stream(self):
        return iter([])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def test_run_turn_stream_sends_cache_control_on_tools_and_last_message(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.messages.stream.return_value = _FakeStreamCtx(_make_response())
    provider._client = fake_client

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"name": "read_file"}]
    original_messages = copy.deepcopy(messages)

    provider.run_turn_stream(messages, "sys", tools, "claude-sonnet-5", on_delta=lambda _: None)

    call_kwargs = fake_client.messages.stream.call_args.kwargs
    assert call_kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert messages == original_messages


# ---------------------------------------------------------------------------
# run_forced_tool: system-prompt caching (previously entirely missing)
# ---------------------------------------------------------------------------


def test_run_forced_tool_caches_the_system_prompt(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    response = MagicMock()
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "extract"  # MagicMock(name=...) would NOT set this -- it's reserved for repr
    tool_use_block.input = {"result": "x"}
    response.content = [tool_use_block]
    fake_client.messages.create.return_value = response
    provider._client = fake_client

    provider.run_forced_tool(
        [{"role": "user", "content": "extract facts"}],
        "extraction instructions",
        {"name": "extract", "description": "d", "input_schema": {}},
        "claude-haiku-4-5",
    )

    call_kwargs = fake_client.messages.create.call_args.kwargs
    # Previously a bare string with no caching at all -- now a cache-controlled block.
    assert call_kwargs["system"] == [
        {"type": "text", "text": "extraction instructions", "cache_control": {"type": "ephemeral"}}
    ]
