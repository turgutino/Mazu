"""Tests for the per-model Anthropic output-token ceiling that replaced the old
flat MAX_TOKENS = 4096 constant. That constant applied to every Anthropic model
regardless of its real capacity -- an artificial bottleneck that could truncate
a long response (a big diff, a long explanation) and make a capable model look
like it produced an incomplete/low-quality answer in a fair cross-provider
benchmark, when the real cause was Mazu's own hardcoded cap.
"""

from unittest.mock import MagicMock, patch

from mazu.llm.capabilities import DEFAULT_ANTHROPIC_MAX_TOKENS, MAX_OUTPUT_TOKENS
from mazu.llm.providers.anthropic_provider import (
    FORCED_TOOL_MAX_TOKENS,
    AnthropicProvider,
    _max_output_tokens,
)


def test_known_model_uses_its_own_table_entry():
    for model, expected in MAX_OUTPUT_TOKENS.items():
        provider_name, model_name = model.split(":", 1)
        if provider_name != "anthropic":
            continue
        assert _max_output_tokens(model_name) == expected


def test_unknown_model_falls_back_to_conservative_default():
    assert _max_output_tokens("claude-some-future-model") == DEFAULT_ANTHROPIC_MAX_TOKENS


def test_default_is_higher_than_the_old_flat_4096_but_not_reckless():
    # The whole point of this change: the fallback must not still be the old
    # bottleneck, but also shouldn't be some arbitrarily huge guessed number.
    assert DEFAULT_ANTHROPIC_MAX_TOKENS > 4096
    assert DEFAULT_ANTHROPIC_MAX_TOKENS <= min(MAX_OUTPUT_TOKENS.values())


def _fake_response():
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = []
    response.usage.model_dump.return_value = {"input_tokens": 1, "output_tokens": 1}
    return response


def test_run_turn_passes_the_model_specific_max_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response()
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "claude-sonnet-5")

    _, kwargs = fake_client.messages.create.call_args
    assert kwargs["max_tokens"] == MAX_OUTPUT_TOKENS["anthropic:claude-sonnet-5"]


def test_run_turn_uses_default_for_an_unlisted_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response()
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "claude-some-future-model")

    _, kwargs = fake_client.messages.create.call_args
    assert kwargs["max_tokens"] == DEFAULT_ANTHROPIC_MAX_TOKENS


def test_run_turn_stream_passes_the_model_specific_max_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_stream_cm = MagicMock()
    fake_stream_cm.__enter__.return_value.text_stream = []
    fake_stream_cm.__enter__.return_value.get_final_message.return_value = _fake_response()
    fake_client.messages.stream.return_value = fake_stream_cm
    provider._client = fake_client

    provider.run_turn_stream(
        [{"role": "user", "content": "hi"}], "sys", [], "claude-haiku-4-5", on_delta=lambda t: None
    )

    _, kwargs = fake_client.messages.stream.call_args
    assert kwargs["max_tokens"] == MAX_OUTPUT_TOKENS["anthropic:claude-haiku-4-5"]


def test_run_forced_tool_keeps_its_own_small_fixed_cap(monkeypatch):
    """run_forced_tool is structured single-field extraction, not a long agentic
    turn -- it must keep its own small constant, independent of the per-model
    table, regardless of which model it's called with."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    response = _fake_response()
    response.content = []
    fake_client.messages.create.return_value = response
    provider._client = fake_client

    tool = {"name": "my_tool", "description": "d", "input_schema": {}}
    provider.run_forced_tool([{"role": "user", "content": "hi"}], "sys", tool, "claude-sonnet-5")

    _, kwargs = fake_client.messages.create.call_args
    assert kwargs["max_tokens"] == FORCED_TOOL_MAX_TOKENS
    assert FORCED_TOOL_MAX_TOKENS != MAX_OUTPUT_TOKENS["anthropic:claude-sonnet-5"]
