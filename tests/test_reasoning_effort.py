"""Tests for the cross-provider "reasoning effort" (low/medium/high) knob.

Mazu's own three-level vocabulary is mapped onto each provider's own native
request shape (verified against the installed SDKs and the claude-api skill's
Thinking & Effort reference -- see each provider's own `_effort_kwargs`/
`_thinking_config` docstring for the exact mapping and its source):
- Anthropic: `thinking: {type: "adaptive"}` + `output_config: {effort: ...}`
- OpenAI (not DeepSeek/local, which share the same base class): `reasoning_effort`
- Gemini: `thinking_config.thinking_level`
"""

from unittest.mock import MagicMock

from mazu.config import set_config_value
from mazu.llm.providers.anthropic_provider import AnthropicProvider
from mazu.llm.providers.deepseek_provider import DeepSeekProvider
from mazu.llm.providers.gemini_provider import GeminiProvider
from mazu.llm.providers.local_provider import LocalProvider
from mazu.llm.providers.openai_provider import OpenAIProvider

# ---------------------------------------------------------------------------
# supports_reasoning_effort flags
# ---------------------------------------------------------------------------


def test_anthropic_and_openai_and_gemini_support_effort():
    assert AnthropicProvider.supports_reasoning_effort is True
    assert OpenAIProvider.supports_reasoning_effort is True
    assert GeminiProvider.supports_reasoning_effort is True


def test_deepseek_and_local_do_not_support_effort():
    # Neither DeepSeek's nor a local OpenAI-compatible server's API has an
    # equivalent parameter -- inherited default from OpenAICompatibleProvider.
    assert DeepSeekProvider.supports_reasoning_effort is False
    assert LocalProvider.supports_reasoning_effort is False


# ---------------------------------------------------------------------------
# supports_reasoning_effort_for -- finer-grained, per-MODEL signal used only by
# mazu-web's effort-chip UI (not consulted by run_turn itself). Real bug found
# live: the class-level flag alone left the chip "active" for openai:gpt-4o (a
# non-reasoning OpenAI model that rejects reasoning_effort with a 400), since
# OpenAI-the-provider does support the parameter for SOME of its own models.
# ---------------------------------------------------------------------------


def test_openai_reasoning_models_report_supported():
    provider = OpenAIProvider()
    for model in ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini", "gpt-5.1-codex"]:
        assert provider.supports_reasoning_effort_for(model) is True, model


def test_openai_non_reasoning_models_report_unsupported():
    provider = OpenAIProvider()
    for model in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo", "chat-latest"]:
        assert provider.supports_reasoning_effort_for(model) is False, model


def test_providers_without_an_override_fall_back_to_the_class_level_flag():
    assert AnthropicProvider().supports_reasoning_effort_for("claude-sonnet-5") is True
    assert GeminiProvider().supports_reasoning_effort_for("gemini-2.0-flash") is True
    assert DeepSeekProvider().supports_reasoning_effort_for("deepseek-chat") is False
    assert LocalProvider().supports_reasoning_effort_for("any-model") is False


# ---------------------------------------------------------------------------
# supports_vision_for -- same best-effort, UI-only, fail-open character as
# supports_reasoning_effort_for above, backing mazu-web's attach-button
# warning. Real bug found live: attaching an image to a plain
# deepseek:deepseek-chat session produced a raw provider error only after
# sending, with nothing warning upfront that the model is text-only.
# ---------------------------------------------------------------------------


def test_deepseek_vision_models_report_supported():
    provider = DeepSeekProvider()
    assert provider.supports_vision_for("deepseek-v4-flash-vision-exp") is True


def test_deepseek_text_only_models_report_unsupported():
    provider = DeepSeekProvider()
    for model in ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"]:
        assert provider.supports_vision_for(model) is False, model


def test_providers_without_a_vision_override_fail_open():
    assert AnthropicProvider().supports_vision_for("claude-sonnet-5") is True
    assert OpenAIProvider().supports_vision_for("gpt-4o") is True
    assert GeminiProvider().supports_vision_for("gemini-2.0-flash") is True
    assert LocalProvider().supports_vision_for("any-model") is True


# ---------------------------------------------------------------------------
# AnthropicProvider -- thinking + output_config.effort
# ---------------------------------------------------------------------------


def _anthropic_response():
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = []
    resp.usage.model_dump.return_value = {}
    return resp


def test_anthropic_run_turn_sends_adaptive_thinking_and_effort(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _anthropic_response()
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "claude-sonnet-5", effort="high")

    _, kwargs = fake_client.messages.create.call_args
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}


def test_anthropic_run_turn_omits_thinking_kwargs_when_effort_unset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _anthropic_response()
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "claude-sonnet-5")

    _, kwargs = fake_client.messages.create.call_args
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


# ---------------------------------------------------------------------------
# OpenAIProvider -- reasoning_effort
# ---------------------------------------------------------------------------


def _openai_response():
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="hi", tool_calls=None))]
    resp.usage.model_dump.return_value = {}
    return resp


def test_openai_run_turn_sends_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    provider = OpenAIProvider()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _openai_response()
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "gpt-5", effort="low")

    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["reasoning_effort"] == "low"


def test_openai_run_turn_omits_reasoning_effort_when_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    provider = OpenAIProvider()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _openai_response()
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "gpt-5")

    _, kwargs = fake_client.chat.completions.create.call_args
    assert "reasoning_effort" not in kwargs


def test_deepseek_run_turn_never_sends_reasoning_effort(monkeypatch):
    # The core reason DeepSeek needs its own test, not just a supports-flag
    # assertion: proves an effort value is silently dropped in the actual
    # request, not merely that the class attribute says it should be.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    provider = DeepSeekProvider()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _openai_response()
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "deepseek-chat", effort="high")

    _, kwargs = fake_client.chat.completions.create.call_args
    assert "reasoning_effort" not in kwargs


# ---------------------------------------------------------------------------
# GeminiProvider -- thinking_config.thinking_level
# ---------------------------------------------------------------------------


def test_gemini_run_turn_sends_thinking_level(monkeypatch):
    from google.genai import types

    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake")
    provider = GeminiProvider()
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.candidates = []
    fake_response.usage_metadata = None
    fake_client.models.generate_content.return_value = fake_response
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "gemini-2.0-flash", effort="medium")

    _, kwargs = fake_client.models.generate_content.call_args
    config = kwargs["config"]
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


def test_gemini_run_turn_omits_thinking_config_when_effort_unset(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake")
    provider = GeminiProvider()
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.candidates = []
    fake_response.usage_metadata = None
    fake_client.models.generate_content.return_value = fake_response
    provider._client = fake_client

    provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "gemini-2.0-flash")

    _, kwargs = fake_client.models.generate_content.call_args
    assert kwargs["config"].thinking_config is None


# ---------------------------------------------------------------------------
# client.default_effort() resolution order
# ---------------------------------------------------------------------------


def test_default_effort_none_when_unset(monkeypatch):
    from mazu.llm.client import default_effort

    monkeypatch.delenv("MAZU_REASONING_EFFORT", raising=False)
    assert default_effort() is None


def test_default_effort_reads_from_config(monkeypatch):
    from mazu.llm.client import default_effort

    monkeypatch.delenv("MAZU_REASONING_EFFORT", raising=False)
    set_config_value("reasoning_effort", "high")
    assert default_effort() == "high"


def test_default_effort_env_var_wins_over_config(monkeypatch):
    from mazu.llm.client import default_effort

    set_config_value("reasoning_effort", "low")
    monkeypatch.setenv("MAZU_REASONING_EFFORT", "high")
    assert default_effort() == "high"


def test_client_run_turn_forwards_resolved_effort_to_provider(monkeypatch):
    from mazu.llm import client as client_module

    monkeypatch.setenv("MAZU_MODEL", "anthropic:claude-sonnet-5")
    monkeypatch.setenv("MAZU_REASONING_EFFORT", "high")
    captured = {}

    def _fake_run_turn(self, messages, system, tools, model, effort=None):
        captured["effort"] = effort
        from mazu.llm.types import AgentResponse

        return AgentResponse(stop_reason="end_turn", content=[], usage={})

    monkeypatch.setattr(AnthropicProvider, "run_turn", _fake_run_turn)
    client_module.run_turn([], "sys", [])

    assert captured["effort"] == "high"
