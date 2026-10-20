"""A missing Anthropic key used to surface as a raw SDK TypeError at request time,
which nothing mapped -- UIs froze instead of showing an auth error."""

import pytest

from mazu.llm.errors import MazuAuthError
from mazu.llm.providers.anthropic_provider import AnthropicProvider


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def test_run_turn_without_key_raises_mazu_auth_error():
    with pytest.raises(MazuAuthError):
        AnthropicProvider().run_turn(
            [{"role": "user", "content": "hi"}], "sys", [], "claude-sonnet-5-5"
        )


def test_unrelated_type_error_is_not_swallowed():
    from mazu.llm.error_mapping import classify_sdk_error
    import anthropic

    with pytest.raises(TypeError, match="boom"):
        classify_sdk_error(anthropic, TypeError("boom"))
