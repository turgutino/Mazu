"""Tests for the streaming seam (run_turn_stream): the base-Provider fallback, the
Anthropic native-streaming implementation, and the OpenAI-compatible implementation's
fragmented tool-call-argument accumulation. All fully mocked -- no network calls.
"""

from unittest.mock import MagicMock

import pytest

from mazu.llm.errors import MazuTransientError
from mazu.llm.providers.anthropic_provider import AnthropicProvider
from mazu.llm.providers.base import Provider
from mazu.llm.providers.deepseek_provider import DeepSeekProvider
from mazu.llm.types import AgentResponse


# ---------------------------------------------------------------------------
# Base Provider fallback (any provider that doesn't override run_turn_stream)
# ---------------------------------------------------------------------------


class _NonStreamingProvider(Provider):
    """Minimal concrete Provider that only implements the required abstract methods,
    to prove the base class's default run_turn_stream() fallback works for any
    provider that never gets a streaming override.
    """

    api_key_env = "FAKE_API_KEY"

    def run_turn(self, messages, system, tools, model) -> AgentResponse:
        return AgentResponse(
            stop_reason="end_turn",
            content=[{"type": "text", "text": "full response text"}],
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    def run_forced_tool(self, messages, system, tool, model) -> dict:
        return {}


def test_default_stream_fallback_delivers_full_text_once():
    provider = _NonStreamingProvider()
    received = []

    response = provider.run_turn_stream(
        [{"role": "user", "content": "hi"}], "sys", [], "some-model", on_delta=received.append
    )

    assert received == ["full response text"]
    assert response.stop_reason == "end_turn"


def test_default_stream_fallback_skips_on_delta_when_no_text():
    class _ToolOnlyProvider(_NonStreamingProvider):
        def run_turn(self, messages, system, tools, model):
            return AgentResponse(
                stop_reason="tool_use",
                content=[{"type": "tool_use", "id": "1", "name": "x", "input": {}}],
                usage={},
            )

    received = []
    _ToolOnlyProvider().run_turn_stream([], "sys", [], "m", on_delta=received.append)
    assert received == []


# ---------------------------------------------------------------------------
# Anthropic: real streaming via client.messages.stream(), reusing
# get_final_message() for identical parsing to the non-streaming path.
# ---------------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self):
        return self._data


class _FakeUsage:
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self):
        return self._data


class _FakeFinalMessage:
    def __init__(self, stop_reason, content_blocks, usage):
        self.stop_reason = stop_reason
        self.content = [_FakeBlock(b) for b in content_blocks]
        self.usage = _FakeUsage(usage)


class _FakeAnthropicStream:
    def __init__(self, deltas, final_message):
        self._deltas = deltas
        self._final = final_message

    @property
    def text_stream(self):
        return iter(self._deltas)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def test_anthropic_stream_delivers_deltas_and_parses_final_message(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()

    final = _FakeFinalMessage(
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "Hello world"}],
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    fake_client = MagicMock()
    fake_client.messages.stream.return_value = _FakeAnthropicStream(["Hello ", "world"], final)
    provider._client = fake_client  # bypass lazy _get_client() construction

    received = []
    response = provider.run_turn_stream(
        [{"role": "user", "content": "hi"}], "sys", [], "claude-sonnet-5", on_delta=received.append
    )

    assert received == ["Hello ", "world"]
    assert response.stop_reason == "end_turn"
    assert response.content == [{"type": "text", "text": "Hello world"}]
    assert response.usage == {"input_tokens": 10, "output_tokens": 5}


def test_anthropic_stream_captures_tool_use_from_final_message(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()

    final = _FakeFinalMessage(
        stop_reason="tool_use",
        content_blocks=[{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}}],
        usage={"input_tokens": 3, "output_tokens": 2},
    )
    fake_client = MagicMock()
    fake_client.messages.stream.return_value = _FakeAnthropicStream([], final)
    provider._client = fake_client

    response = provider.run_turn_stream([], "sys", [], "claude-sonnet-5", on_delta=lambda _: None)

    assert response.stop_reason == "tool_use"
    assert response.content[0]["name"] == "read_file"


# ---------------------------------------------------------------------------
# OpenAI-compatible (openai:/deepseek:): fragmented tool-call argument deltas
# must be accumulated by index before JSON-parsing.
# ---------------------------------------------------------------------------


class _FakeFunctionDelta:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _FakeToolCallDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _FakeFunctionDelta(name, arguments)


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


def _make_provider_with_stream(monkeypatch, chunks):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    provider = DeepSeekProvider()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(chunks)
    provider._client = fake_client
    return provider


def test_openai_compatible_stream_text_only(monkeypatch):
    chunks = [
        _FakeChunk(choices=[_FakeChoice(_FakeDelta(content="Hello "))]),
        _FakeChunk(choices=[_FakeChoice(_FakeDelta(content="world"))]),
        _FakeChunk(choices=[], usage=_FakeUsage({"prompt_tokens": 8, "completion_tokens": 2})),
    ]
    provider = _make_provider_with_stream(monkeypatch, chunks)

    received = []
    response = provider.run_turn_stream(
        [{"role": "user", "content": "hi"}], "sys", [], "deepseek-chat", on_delta=received.append
    )

    assert received == ["Hello ", "world"]
    assert response.stop_reason == "end_turn"
    assert response.content == [{"type": "text", "text": "Hello world"}]
    assert response.usage == {"prompt_tokens": 8, "completion_tokens": 2}


def test_openai_compatible_stream_accumulates_fragmented_tool_call_arguments(monkeypatch):
    # A realistic wire pattern: id+name arrive on the first delta for an index, then
    # the JSON arguments string trickles in across several further deltas.
    chunks = [
        _FakeChunk(
            choices=[
                _FakeChoice(
                    _FakeDelta(tool_calls=[_FakeToolCallDelta(index=0, id="call_1", name="read_file", arguments="")])
                )
            ]
        ),
        _FakeChunk(choices=[_FakeChoice(_FakeDelta(tool_calls=[_FakeToolCallDelta(index=0, arguments='{"pa')]))]),
        _FakeChunk(choices=[_FakeChoice(_FakeDelta(tool_calls=[_FakeToolCallDelta(index=0, arguments='th": ')]))]),
        _FakeChunk(choices=[_FakeChoice(_FakeDelta(tool_calls=[_FakeToolCallDelta(index=0, arguments='"a.py"}')]))]),
        _FakeChunk(choices=[], usage=_FakeUsage({"prompt_tokens": 5, "completion_tokens": 4})),
    ]
    provider = _make_provider_with_stream(monkeypatch, chunks)

    response = provider.run_turn_stream([], "sys", [], "deepseek-chat", on_delta=lambda _: None)

    assert response.stop_reason == "tool_use"
    assert response.content == [
        {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py"}}
    ]


def test_openai_compatible_stream_multiple_parallel_tool_calls_kept_separate(monkeypatch):
    chunks = [
        _FakeChunk(
            choices=[
                _FakeChoice(
                    _FakeDelta(
                        tool_calls=[
                            _FakeToolCallDelta(index=0, id="call_a", name="read_file", arguments='{"path": "a.py"}'),
                            _FakeToolCallDelta(index=1, id="call_b", name="read_file", arguments='{"path": "b.py"}'),
                        ]
                    )
                )
            ]
        ),
        _FakeChunk(choices=[]),
    ]
    provider = _make_provider_with_stream(monkeypatch, chunks)

    response = provider.run_turn_stream([], "sys", [], "deepseek-chat", on_delta=lambda _: None)

    assert len(response.content) == 2
    assert response.content[0]["input"] == {"path": "a.py"}
    assert response.content[1]["input"] == {"path": "b.py"}


# ---------------------------------------------------------------------------
# client.py's run_turn_stream(): must NOT auto-retry (a retry after partial
# deltas were already printed would duplicate visible output).
# ---------------------------------------------------------------------------


def test_client_run_turn_stream_does_not_retry_on_transient_error(monkeypatch):
    from mazu.llm import client as client_module

    monkeypatch.setenv("MAZU_MODEL", "deepseek:deepseek-chat")
    call_count = {"n": 0}

    def _fake_run_turn_stream(self, messages, system, tools, model, on_delta):
        call_count["n"] += 1
        on_delta("partial output before failure")
        raise MazuTransientError("connection dropped mid-stream")

    monkeypatch.setattr(DeepSeekProvider, "run_turn_stream", _fake_run_turn_stream)

    with pytest.raises(MazuTransientError):
        client_module.run_turn_stream([], "sys", [], on_delta=lambda _: None)

    assert call_count["n"] == 1  # exactly once -- no retry
