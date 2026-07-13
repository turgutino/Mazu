import os

import pytest

from mazu.config import (
    ensure_api_key,
    get_default_model,
    list_config,
    load_config,
    set_config_value,
    unset_config_value,
)


def test_missing_key_shows_set_syntax_on_windows(monkeypatch):
    monkeypatch.setattr("mazu.config.platform.system", lambda: "Windows")
    # Re-import to pick up the module-level _SET_ENV_EXAMPLE computed from
    # platform.system() at import time -- patch it directly instead, matching
    # what the code actually reads.
    monkeypatch.setattr("mazu.config._SET_ENV_EXAMPLE", "set {var}=...")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("MAZU_MODEL", "deepseek:deepseek-chat")

    with pytest.raises(SystemExit) as exc_info:
        ensure_api_key()

    message = str(exc_info.value)
    assert "set DEEPSEEK_API_KEY=..." in message
    assert "export" not in message


def test_missing_key_shows_export_syntax_on_posix(monkeypatch):
    monkeypatch.setattr("mazu.config._SET_ENV_EXAMPLE", "export {var}=...")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("MAZU_MODEL", "deepseek:deepseek-chat")

    with pytest.raises(SystemExit) as exc_info:
        ensure_api_key()

    message = str(exc_info.value)
    assert "export DEEPSEEK_API_KEY=..." in message


def test_key_present_does_not_raise(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setenv("MAZU_MODEL", "deepseek:deepseek-chat")
    ensure_api_key()  # should not raise


# ---------------------------------------------------------------------------
# persistent config: set / list / unset / load
# ---------------------------------------------------------------------------
# Isolation from the real ~/.mazu/config.toml is handled globally by conftest.py's
# autouse _isolate_mazu_config_path fixture -- every test in this file (and every
# other test in the suite) already gets its own tmp_path-backed config file.


def test_list_config_empty_when_no_file():
    assert list_config() == {}


def test_set_and_list_round_trips_a_value():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    assert list_config() == {"default_model": "anthropic:claude-opus-4-8"}


def test_set_unknown_key_raises_value_error():
    with pytest.raises(ValueError, match="Unknown config key"):
        set_config_value("bogus_key", "x")


def test_set_overwrites_existing_value():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    set_config_value("default_model", "deepseek:deepseek-chat")
    assert list_config()["default_model"] == "deepseek:deepseek-chat"


def test_set_preserves_other_keys():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    set_config_value("anthropic_api_key", "sk-ant-123")
    values = list_config()
    assert values["default_model"] == "anthropic:claude-opus-4-8"
    assert values["anthropic_api_key"] == "sk-ant-123"


def test_unset_removes_a_value():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    ok = unset_config_value("default_model")
    assert ok is True
    assert list_config() == {}


def test_unset_missing_key_returns_false_and_changes_nothing():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    ok = unset_config_value("nonexistent_key")
    assert ok is False
    assert list_config() == {"default_model": "anthropic:claude-opus-4-8"}


def test_set_value_with_special_characters_round_trips_through_toml(tmp_path):
    # Backslashes and quotes must not corrupt the written TOML or break re-parsing.
    tricky = 'sk-"quoted"-and-\\backslash\\-value'
    set_config_value("anthropic_api_key", tricky)
    assert list_config()["anthropic_api_key"] == tricky


def test_get_default_model_returns_none_when_unset():
    assert get_default_model() is None


def test_get_default_model_returns_configured_value():
    set_config_value("default_model", "deepseek:deepseek-chat")
    assert get_default_model() == "deepseek:deepseek-chat"


def test_load_config_malformed_toml_warns_and_returns_empty(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text("this is not valid toml [[[", encoding="utf-8")
    result = load_config()
    assert result == {}
    assert "malformed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# API key injection into the environment
# ---------------------------------------------------------------------------


def test_load_config_injects_legacy_api_key_as_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    set_config_value("api_key", "sk-legacy-anthropic")
    load_config()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-legacy-anthropic"


def test_load_config_injects_per_provider_keys(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    set_config_value("deepseek_api_key", "sk-deepseek-123")
    load_config()
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-deepseek-123"


def test_load_config_never_overwrites_an_existing_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real-env-value")
    set_config_value("api_key", "sk-config-file-value")
    load_config()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-real-env-value"


# ---------------------------------------------------------------------------
# default_model() precedence (client.py, config-file layer)
# ---------------------------------------------------------------------------


def test_default_model_uses_config_when_no_mazu_model_env(monkeypatch):
    from mazu.llm.client import default_model

    monkeypatch.delenv("MAZU_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    set_config_value("default_model", "deepseek:deepseek-chat")

    assert default_model() == "deepseek:deepseek-chat"


def test_default_model_mazu_model_env_wins_over_config(monkeypatch):
    from mazu.llm.client import default_model

    monkeypatch.setenv("MAZU_MODEL", "openai:gpt-5")
    set_config_value("default_model", "deepseek:deepseek-chat")

    assert default_model() == "openai:gpt-5"


def test_default_model_falls_back_to_auto_detect_when_config_unset(monkeypatch):
    from mazu.llm.client import default_model

    monkeypatch.delenv("MAZU_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert default_model() == "deepseek:deepseek-chat"
