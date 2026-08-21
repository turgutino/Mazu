import os

import pytest

import mazu.curator.client as curator_client_module
from mazu.config import ensure_api_key, set_config_value
from mazu.curator.client import CuratorNotConfigured, resolve_curator_provider
from mazu.llm.client import _PROVIDERS


def test_raises_when_unconfigured():
    with pytest.raises(CuratorNotConfigured):
        resolve_curator_provider()


def test_resolves_anthropic_provider_with_isolated_key(monkeypatch):
    curator_client_module._providers.clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")

    provider, model_name = resolve_curator_provider()

    assert model_name == "claude-haiku-4-5"
    assert provider.api_key_env == "MAZU_CURATOR_API_KEY"


def test_resolves_deepseek_provider(monkeypatch):
    curator_client_module._providers.clear()
    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "deepseek:deepseek-chat")

    provider, model_name = resolve_curator_provider()

    assert model_name == "deepseek-chat"
    assert provider.api_key_env == "MAZU_CURATOR_API_KEY"
    assert provider.base_url == "https://api.deepseek.com"


def test_shared_providers_singleton_dict_is_never_touched(monkeypatch):
    """The actual isolation invariant: resolving Curator's own provider must never
    mutate or add to mazu.llm.client._PROVIDERS -- the identities of the main
    model's singleton provider instances must be byte-for-byte unchanged."""
    curator_client_module._providers.clear()
    before = {name: id(p) for name, p in _PROVIDERS.items()}
    before_keys = set(_PROVIDERS.keys())

    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")
    resolve_curator_provider()

    after = {name: id(p) for name, p in _PROVIDERS.items()}
    assert before == after
    assert set(_PROVIDERS.keys()) == before_keys


def test_main_ensure_api_key_still_fails_with_only_curator_configured(monkeypatch):
    """Decisive end-to-end proof: a fully configured Curator must not make the
    main model appear to have a key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("MAZU_MODEL", "anthropic:claude-sonnet-5")
    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")

    with pytest.raises(SystemExit):
        ensure_api_key()


def test_curator_env_var_does_not_leak_into_main_key_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    curator_client_module._providers.clear()
    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")

    resolve_curator_provider()

    assert os.environ.get("ANTHROPIC_API_KEY") is None
