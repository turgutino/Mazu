"""Addendum 8: local model support (LM Studio, Ollama, any OpenAI-compatible local
server). Config isolation from the real ~/.mazu/config.toml is handled globally by
conftest.py's autouse _isolate_mazu_config_path fixture.
"""

import pytest

from mazu.config import ensure_api_key, local_base_url, set_config_value, unset_config_value
from mazu.llm.client import _PROVIDER_DEFAULT_MODELS, _PROVIDER_PRIORITY, _PROVIDERS, default_model
from mazu.llm.providers.local_provider import LocalProvider


# ---------------------------------------------------------------------------
# requires_api_key opt-out
# ---------------------------------------------------------------------------


def test_ensure_api_key_skips_check_for_local_model(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "MAZU_LOCAL_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    ensure_api_key("local:llama3.1")  # should not raise


def test_ensure_api_key_still_raises_for_missing_cloud_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        ensure_api_key("anthropic:claude-sonnet-5")


def test_local_provider_requires_api_key_is_false():
    assert LocalProvider().requires_api_key is False


def test_cloud_providers_still_require_api_key_by_default():
    assert _PROVIDERS["anthropic"].requires_api_key is True
    assert _PROVIDERS["openai"].requires_api_key is True
    assert _PROVIDERS["deepseek"].requires_api_key is True


# ---------------------------------------------------------------------------
# _get_client() placeholder key
# ---------------------------------------------------------------------------


def test_get_client_uses_placeholder_key_when_unset(monkeypatch):
    monkeypatch.delenv("MAZU_LOCAL_API_KEY", raising=False)
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    provider = LocalProvider()
    provider._get_client()

    assert captured["api_key"] == "not-needed"
    assert captured["base_url"] == "http://localhost:1234/v1"


def test_get_client_uses_real_value_when_set(monkeypatch):
    monkeypatch.setenv("MAZU_LOCAL_API_KEY", "real-token")
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    LocalProvider()._get_client()
    assert captured["api_key"] == "real-token"


# ---------------------------------------------------------------------------
# local_base_url() precedence
# ---------------------------------------------------------------------------


def test_local_base_url_defaults_to_lm_studio_port(monkeypatch):
    monkeypatch.delenv("MAZU_LOCAL_BASE_URL", raising=False)
    assert local_base_url() == "http://localhost:1234/v1"


def test_local_base_url_config_value_wins_over_default(monkeypatch):
    monkeypatch.delenv("MAZU_LOCAL_BASE_URL", raising=False)
    set_config_value("local_base_url", "http://localhost:11434/v1")
    assert local_base_url() == "http://localhost:11434/v1"


def test_local_base_url_env_wins_over_config(monkeypatch):
    set_config_value("local_base_url", "http://localhost:11434/v1")
    monkeypatch.setenv("MAZU_LOCAL_BASE_URL", "http://192.168.1.5:1234/v1")
    assert local_base_url() == "http://192.168.1.5:1234/v1"


def test_local_base_url_config_set_list_unset_round_trip():
    set_config_value("local_base_url", "http://localhost:11434/v1")
    ok = unset_config_value("local_base_url")
    assert ok is True


# ---------------------------------------------------------------------------
# Lazy base_url resolution
# ---------------------------------------------------------------------------


def test_base_url_not_resolved_until_get_client_called():
    provider = LocalProvider()
    assert provider.base_url is None  # not resolved at construction time


def test_base_url_resolved_lazily_on_first_get_client(monkeypatch):
    monkeypatch.setenv("MAZU_LOCAL_BASE_URL", "http://localhost:9999/v1")

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    provider = LocalProvider()
    provider._get_client()
    assert provider.base_url == "http://localhost:9999/v1"


# ---------------------------------------------------------------------------
# No silent auto-selection
# ---------------------------------------------------------------------------


def test_local_never_in_provider_priority():
    assert "local" not in _PROVIDER_PRIORITY


def test_default_model_never_auto_selects_local(monkeypatch):
    monkeypatch.delenv("MAZU_MODEL", raising=False)
    monkeypatch.setenv("MAZU_LOCAL_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert default_model() == "anthropic:claude-sonnet-5"


def test_local_has_a_default_model_entry_for_mazu_models():
    assert _PROVIDER_DEFAULT_MODELS["local"] == "local:your-model-name"


# ---------------------------------------------------------------------------
# mazu models row
# ---------------------------------------------------------------------------


def test_mazu_models_shows_a_local_row():
    from mazu.llm.capabilities import list_capabilities

    rows = list_capabilities()
    local_rows = [r for r in rows if r.provider == "local"]
    assert len(local_rows) == 1
    row = local_rows[0]
    assert row.streaming is True
    assert row.tool_use is True
    assert row.context_window is None
    assert row.input_price_per_million is None
    assert row.output_price_per_million is None


# ---------------------------------------------------------------------------
# _split_model regression (multi-colon Ollama-style tags)
# ---------------------------------------------------------------------------


def test_split_model_handles_ollama_style_multi_colon_tag():
    from mazu.llm.client import _split_model

    assert _split_model("local:llama3.1:8b") == ("local", "llama3.1:8b")
