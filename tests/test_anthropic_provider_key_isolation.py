"""Regression test for a real, load-bearing bug found while designing the Curator
feature: AnthropicProvider._get_client() read self.api_key_env only to sanity-check
it was ASCII, then constructed `Anthropic()` with no `api_key=` argument at all --
so the SDK silently fell back to reading ANTHROPIC_API_KEY directly, ignoring
self.api_key_env entirely. Retargeting api_key_env to a different env var (the
whole point of letting a caller use a separate key) would silently keep using
whatever ANTHROPIC_API_KEY happened to be set to instead -- and it would fail
*silently*, not raise, since the call would still succeed, just billed/authed
against the wrong key.
"""

from unittest.mock import MagicMock, patch

import pytest

from mazu.llm.errors import MazuAuthError
from mazu.llm.providers.anthropic_provider import AnthropicProvider


def test_retargeted_api_key_env_is_actually_used(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MAZU_CURATOR_API_KEY", "sk-curator-key")

    provider = AnthropicProvider()
    provider.api_key_env = "MAZU_CURATOR_API_KEY"

    with patch("anthropic.Anthropic") as mock_anthropic:
        provider._get_client()

    mock_anthropic.assert_called_once_with(api_key="sk-curator-key")


def test_default_behavior_unchanged_when_using_the_standard_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-main-key")

    provider = AnthropicProvider()

    with patch("anthropic.Anthropic") as mock_anthropic:
        provider._get_client()

    mock_anthropic.assert_called_once_with(api_key="sk-main-key")


def test_retargeted_env_var_does_not_leak_the_main_key(monkeypatch):
    """The actual isolation guarantee: even if the user's own ANTHROPIC_API_KEY is
    set, a provider retargeted at a different env var must never pick it up."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-main-key")
    monkeypatch.setenv("MAZU_CURATOR_API_KEY", "sk-curator-key")

    provider = AnthropicProvider()
    provider.api_key_env = "MAZU_CURATOR_API_KEY"

    with patch("anthropic.Anthropic") as mock_anthropic:
        provider._get_client()

    mock_anthropic.assert_called_once_with(api_key="sk-curator-key")


def test_non_ascii_key_still_raises_clean_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sənin-key-in")
    provider = AnthropicProvider()
    with pytest.raises(MazuAuthError, match="non-ASCII"):
        provider._get_client()


def test_missing_key_falls_back_to_none_not_a_crash(monkeypatch):
    """No key set at all -- api_key=None is passed through explicitly, matching the
    SDK's own documented fallback behavior (it does its own env lookup when given
    None), so this must not raise here or change the caller-visible failure mode."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider()

    with patch("anthropic.Anthropic") as mock_anthropic:
        provider._get_client()

    mock_anthropic.assert_called_once_with(api_key=None)
