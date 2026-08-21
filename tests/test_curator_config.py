from mazu.config import set_config_value
from mazu.curator.config import (
    curator_api_key,
    curator_base_url,
    curator_configured,
    curator_enabled,
    curator_model,
)


def test_unconfigured_by_default():
    assert curator_configured() is False
    assert curator_api_key() is None
    assert curator_model() is None


def test_configured_once_both_key_and_model_are_set():
    set_config_value("curator_api_key", "sk-curator")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")
    assert curator_configured() is True


def test_key_alone_is_not_enough():
    set_config_value("curator_api_key", "sk-curator")
    assert curator_configured() is False


def test_model_alone_is_not_enough():
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")
    assert curator_configured() is False


def test_env_var_wins_over_config_for_api_key(monkeypatch):
    set_config_value("curator_api_key", "sk-from-config")
    monkeypatch.setenv("MAZU_CURATOR_API_KEY", "sk-from-env")
    assert curator_api_key() == "sk-from-env"


def test_env_var_wins_over_config_for_model(monkeypatch):
    set_config_value("curator_model", "deepseek:deepseek-chat")
    monkeypatch.setenv("MAZU_CURATOR_MODEL", "anthropic:claude-opus-4-8")
    assert curator_model() == "anthropic:claude-opus-4-8"


def test_curator_key_never_falls_back_to_main_provider_env_vars(monkeypatch):
    """The core isolation guarantee at the config layer: even with a real
    ANTHROPIC_API_KEY set, curator_api_key() must never return it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-main-key")
    monkeypatch.delenv("MAZU_CURATOR_API_KEY", raising=False)
    assert curator_api_key() is None


def test_enabled_defaults_true_once_configured():
    set_config_value("curator_api_key", "sk-curator")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")
    assert curator_enabled() is True


def test_disable_flips_enabled_without_clearing_config():
    set_config_value("curator_api_key", "sk-curator")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")
    set_config_value("curator_enabled", "false")
    assert curator_enabled() is False
    assert curator_configured() is True  # config itself is untouched


def test_base_url_only_relevant_for_local_default_none():
    assert curator_base_url() is None
    set_config_value("curator_base_url", "http://localhost:9999/v1")
    assert curator_base_url() == "http://localhost:9999/v1"
