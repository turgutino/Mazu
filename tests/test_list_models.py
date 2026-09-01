"""Tests for Provider.list_models() -- the live "which models can this key
actually use" catalog call that backs mazu-web's model picker (see
mazu_web/app.py's /api/models/<provider>/live), distinct from the static
capability table in mazu/llm/capabilities.py.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from anthropic import AnthropicError
from anthropic.types.model_info import ModelInfo
from openai import OpenAIError
from openai.types.model import Model as OpenAIModel

from mazu.llm.client import list_models
from mazu.llm.errors import MazuAPIError
from mazu.llm.providers.anthropic_provider import AnthropicProvider
from mazu.llm.providers.deepseek_provider import DeepSeekProvider
from mazu.llm.providers.local_provider import LocalProvider
from mazu.llm.providers.openai_provider import OpenAIProvider

# ---------------------------------------------------------------------------
# AnthropicProvider.list_models()
# ---------------------------------------------------------------------------


def _anthropic_model(model_id: str) -> ModelInfo:
    return ModelInfo(
        id=model_id, created_at=datetime.now(timezone.utc), display_name=model_id, type="model"
    )


def test_anthropic_list_models_returns_ids_sorted_descending(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.models.list.return_value = [
        _anthropic_model("claude-sonnet-5"),
        _anthropic_model("claude-opus-4-8"),
    ]
    provider._client = fake_client

    assert provider.list_models() == ["claude-sonnet-5", "claude-opus-4-8"]


def test_anthropic_list_models_wraps_sdk_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    provider = AnthropicProvider()
    fake_client = MagicMock()
    fake_client.models.list.side_effect = AnthropicError("bad key")
    provider._client = fake_client

    with pytest.raises(MazuAPIError):
        provider.list_models()


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.list_models() -- via DeepSeekProvider and
# LocalProvider, neither of which filters the catalog at all.
# ---------------------------------------------------------------------------


def _openai_model(model_id: str) -> OpenAIModel:
    return OpenAIModel(id=model_id, created=0, object="model", owned_by="test")


def test_deepseek_list_models_returns_full_catalog_unfiltered(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    provider = DeepSeekProvider()
    fake_client = MagicMock()
    fake_client.models.list.return_value = MagicMock(
        data=[_openai_model("deepseek-reasoner"), _openai_model("deepseek-chat")]
    )
    provider._client = fake_client

    assert provider.list_models() == ["deepseek-chat", "deepseek-reasoner"]


def test_deepseek_list_models_wraps_sdk_errors(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    provider = DeepSeekProvider()
    fake_client = MagicMock()
    fake_client.models.list.side_effect = OpenAIError("bad key")
    provider._client = fake_client

    with pytest.raises(MazuAPIError):
        provider.list_models()


def test_local_provider_list_models_also_unfiltered(monkeypatch):
    monkeypatch.delenv("MAZU_LOCAL_API_KEY", raising=False)
    provider = LocalProvider()
    fake_client = MagicMock()
    fake_client.models.list.return_value = MagicMock(data=[_openai_model("llama3.1:8b")])
    provider._client = fake_client

    assert provider.list_models() == ["llama3.1:8b"]


# ---------------------------------------------------------------------------
# OpenAIProvider._is_chat_model() filtering -- OpenAI's own /v1/models mixes
# embeddings/TTS/image/moderation/legacy-completion ids in with real chat models.
# ---------------------------------------------------------------------------


def test_openai_list_models_filters_out_non_chat_families(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    provider = OpenAIProvider()
    fake_client = MagicMock()
    fake_client.models.list.return_value = MagicMock(
        data=[
            _openai_model("gpt-5"),
            _openai_model("gpt-4o-mini"),
            _openai_model("text-embedding-3-large"),
            _openai_model("whisper-1"),
            _openai_model("tts-1"),
            _openai_model("dall-e-3"),
            _openai_model("text-moderation-latest"),
            _openai_model("davinci-002"),
        ]
    )
    provider._client = fake_client

    assert provider.list_models() == ["gpt-4o-mini", "gpt-5"]


# ---------------------------------------------------------------------------
# mazu.llm.client.list_models() dispatcher
# ---------------------------------------------------------------------------


def test_client_list_models_dispatches_to_the_right_provider(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    from mazu.llm.client import _PROVIDERS

    fake_client = MagicMock()
    fake_client.models.list.return_value = MagicMock(data=[_openai_model("deepseek-chat")])
    _PROVIDERS["deepseek"]._client = fake_client
    try:
        assert list_models("deepseek") == ["deepseek-chat"]
    finally:
        _PROVIDERS["deepseek"]._client = None


def test_client_list_models_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        list_models("not-a-real-provider")


def test_client_list_models_raises_clean_auth_error_when_key_missing(monkeypatch):
    # Real bug found via live testing: AnthropicProvider._get_client() passes a
    # missing key through to the SDK as None on purpose (its own client
    # construction must not raise here -- see
    # test_anthropic_provider_key_isolation.py), so without this guard a totally
    # unset ANTHROPIC_API_KEY surfaced as a raw SDK TypeError deep in HTTP header
    # construction (500 in mazu-web) instead of a clean, catchable error.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mazu.llm.errors import MazuAuthError

    with pytest.raises(MazuAuthError, match="ANTHROPIC_API_KEY"):
        list_models("anthropic")


def test_client_list_models_propagates_sdk_errors(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mazu.llm.client import _PROVIDERS

    fake_client = MagicMock()
    fake_client.models.list.side_effect = AnthropicError("no key")
    _PROVIDERS["anthropic"]._client = fake_client
    try:
        with pytest.raises(MazuAPIError):
            list_models("anthropic")
    finally:
        _PROVIDERS["anthropic"]._client = None
