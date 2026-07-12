import sys

import pytest

from mazu.llm.client import _PROVIDERS, _split_model, default_model
from mazu.llm.errors import MazuAuthError
from mazu.llm.providers.deepseek_provider import DeepSeekProvider


def test_split_model_with_provider_prefix():
    assert _split_model("openai:gpt-5") == ("openai", "gpt-5")
    assert _split_model("deepseek:deepseek-chat") == ("deepseek", "deepseek-chat")


def test_split_model_bare_name_assumes_anthropic():
    assert _split_model("claude-opus-4-8") == ("anthropic", "claude-opus-4-8")


def test_all_four_providers_registered():
    assert set(_PROVIDERS) == {"anthropic", "openai", "deepseek", "gemini"}


def test_default_model_prefers_explicit_env_var(monkeypatch):
    monkeypatch.setenv("MAZU_MODEL", "deepseek:deepseek-reasoner")
    assert default_model() == "deepseek:deepseek-reasoner"


def test_default_model_auto_detects_from_available_key(monkeypatch):
    monkeypatch.delenv("MAZU_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    assert default_model() == "deepseek:deepseek-chat"


def test_default_model_anthropic_is_tiebreaker(monkeypatch):
    monkeypatch.delenv("MAZU_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    assert default_model() == "anthropic:claude-sonnet-5"


def test_missing_api_key_raises_clean_auth_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = DeepSeekProvider()
    with pytest.raises(MazuAuthError, match="DEEPSEEK_API_KEY"):
        provider._get_client()


def test_non_ascii_api_key_raises_clean_auth_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sənin-key-in")  # placeholder pasted by mistake
    provider = DeepSeekProvider()
    with pytest.raises(MazuAuthError, match="non-ASCII"):
        provider._get_client()


def test_missing_openai_package_raises_clean_error_not_module_not_found(monkeypatch):
    """Regression test for a real bug found via live testing: run_turn() used to do an
    unguarded `import openai` before _get_client()'s guarded one ever ran, so a missing
    `openai` package crashed with a raw ModuleNotFoundError instead of a clean message.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setitem(sys.modules, "openai", None)  # forces `import openai` to raise ImportError

    provider = DeepSeekProvider()
    with pytest.raises(MazuAuthError, match="openai package isn't installed"):
        provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "deepseek-chat")


def test_missing_openai_package_raises_clean_error_for_forced_tool(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setitem(sys.modules, "openai", None)

    provider = DeepSeekProvider()
    with pytest.raises(MazuAuthError, match="openai package isn't installed"):
        provider.run_forced_tool(
            [{"role": "user", "content": "hi"}],
            "sys",
            {"name": "x", "description": "d", "input_schema": {}},
            "deepseek-chat",
        )
