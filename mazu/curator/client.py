"""Curator's own LLM client -- deliberately NOT mazu.llm.client.run_turn/_PROVIDERS.
Those cache singleton Provider instances that read their key from the MAIN
provider env vars (ANTHROPIC_API_KEY, etc.); reusing them for Curator would
silently authenticate Curator's calls with the user's own key instead of
curator_api_key -- exactly the failure the whole separate-key design exists to
prevent. This module builds fresh Provider instances retargeted at a distinct env
var (MAZU_CURATOR_API_KEY) that no entry in mazu.llm.client._PROVIDERS ever reads,
so the two paths cannot structurally cross. See the Curator plan's Phase 0 note:
this only works correctly because AnthropicProvider now actually passes api_key=
through to the SDK instead of relying on the SDK's own env fallback.
"""
import os

from mazu.curator.config import CURATOR_API_KEY_ENV, curator_api_key, curator_base_url, curator_model
from mazu.llm.client import _split_model
from mazu.llm.retry import with_retry
from mazu.llm.types import AgentResponse

# One provider instance per provider name, built lazily and cached for the life of
# this process (each `mazu curator run` invocation is its own short-lived process,
# so there's no need to handle a key changing mid-run).
_providers: dict[str, object] = {}


class CuratorNotConfigured(RuntimeError):
    pass


def _build_provider(provider_name: str):
    if provider_name == "anthropic":
        from mazu.llm.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        provider.api_key_env = CURATOR_API_KEY_ENV
        return provider
    if provider_name == "gemini":
        from mazu.llm.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider()
        provider.api_key_env = CURATOR_API_KEY_ENV
        return provider
    if provider_name == "openai":
        from mazu.llm.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            base_url=None, api_key_env=CURATOR_API_KEY_ENV, provider_label="curator:openai:"
        )
    if provider_name == "deepseek":
        from mazu.llm.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            base_url="https://api.deepseek.com",
            api_key_env=CURATOR_API_KEY_ENV,
            provider_label="curator:deepseek:",
        )
    if provider_name == "local":
        from mazu.llm.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            base_url=curator_base_url() or "http://localhost:1234/v1",
            api_key_env=CURATOR_API_KEY_ENV,
            provider_label="curator:local:",
            requires_api_key=False,
        )
    raise ValueError(
        f"Unknown provider '{provider_name}' in curator_model. "
        "Available: anthropic, openai, deepseek, gemini, local."
    )


def resolve_curator_provider() -> tuple[object, str]:
    """Returns (provider, model_name). Raises CuratorNotConfigured if either
    curator_api_key or curator_model is unset -- callers use this (or
    curator.config.curator_configured()) to guarantee zero API calls when
    Curator hasn't been set up."""
    model = curator_model()
    key = curator_api_key()
    if model is None or key is None:
        raise CuratorNotConfigured(
            "Curator is not configured. Run `mazu curator setup`, or set both "
            "curator_model and curator_api_key (mazu config set / MAZU_CURATOR_API_KEY)."
        )
    # Set once per process before first client construction -- the provider's
    # _get_client() reads this env var lazily on first use.
    os.environ[CURATOR_API_KEY_ENV] = key
    provider_name, model_name = _split_model(model)
    if provider_name not in _providers:
        _providers[provider_name] = _build_provider(provider_name)
    return _providers[provider_name], model_name


def run_curator_turn(messages: list[dict], system: str, tools: list[dict]) -> AgentResponse:
    """The single seam through which every Curator LLM call flows -- mirrors
    mazu.llm.client.run_turn's shape exactly, but resolves against Curator's own
    isolated provider/key instead of the shared _PROVIDERS singletons."""
    provider, model_name = resolve_curator_provider()
    return with_retry(lambda: provider.run_turn(messages, system, tools, model_name))
