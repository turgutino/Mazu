import os
from typing import Callable

from mazu.llm.providers.anthropic_provider import AnthropicProvider
from mazu.llm.providers.deepseek_provider import DeepSeekProvider
from mazu.llm.providers.gemini_provider import GeminiProvider
from mazu.llm.providers.openai_provider import OpenAIProvider
from mazu.llm.retry import with_retry
from mazu.llm.types import AgentResponse

_PROVIDERS = {
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
    "deepseek": DeepSeekProvider(),
    "gemini": GeminiProvider(),
}

# Tie-breaker order only — used when nothing is configured and we have to guess among
# multiple keys that happen to be set. Not a requirement to have any specific one.
# Gemini appended at the end so it never changes default resolution for existing setups.
_PROVIDER_PRIORITY = ["anthropic", "openai", "deepseek", "gemini"]

_PROVIDER_DEFAULT_MODELS = {
    "anthropic": "anthropic:claude-sonnet-5",
    "openai": "openai:gpt-5",
    "deepseek": "deepseek:deepseek-chat",
    "gemini": "gemini:gemini-2.0-flash",
}


def default_model() -> str:
    """Resolution order: explicit MAZU_MODEL env var, then auto-detect from whichever
    provider's API key is actually present in the environment, then a hardcoded
    Anthropic fallback (which will simply surface a clear "set ANTHROPIC_API_KEY" error
    via config.ensure_api_key if the user genuinely has no key configured at all).
    Nothing here assumes Anthropic is required — a DeepSeek-only or OpenAI-only setup
    is auto-detected and used with no extra flags needed.
    """
    if os.environ.get("MAZU_MODEL"):
        return os.environ["MAZU_MODEL"]
    for provider_name in _PROVIDER_PRIORITY:
        if os.environ.get(_PROVIDERS[provider_name].api_key_env):
            return _PROVIDER_DEFAULT_MODELS[provider_name]
    return _PROVIDER_DEFAULT_MODELS["anthropic"]


def _split_model(model: str) -> tuple[str, str]:
    """'openai:gpt-5' -> ('openai', 'gpt-5'). A bare model name with no ':' is assumed
    to be Anthropic, so existing MAZU_MODEL=claude-opus-4-8-style config keeps working.
    """
    if ":" in model:
        provider, name = model.split(":", 1)
        return provider, name
    return "anthropic", model


def _resolve_provider(model: str | None) -> tuple[str, object, str]:
    provider_name, model_name = _split_model(model or default_model())
    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        raise ValueError(
            f"Unknown provider '{provider_name}'. Available: {', '.join(_PROVIDERS)}"
        )
    return provider_name, provider, model_name


def run_turn(
    messages: list[dict],
    system: str,
    tools: list[dict],
    model: str | None = None,
) -> AgentResponse:
    """The single seam through which every LLM call in Mazu flows. Adding a new
    provider means writing one more Provider subclass in mazu/llm/providers/ and
    registering it above — callers never change. Transient failures (timeouts,
    rate limits) are retried with backoff here; everything else (auth, bad
    request, context-length) surfaces immediately as a MazuAPIError subclass for
    the caller to handle.
    """
    _, provider, model_name = _resolve_provider(model)
    return with_retry(lambda: provider.run_turn(messages, system, tools, model_name))


def run_turn_stream(
    messages: list[dict],
    system: str,
    tools: list[dict],
    on_delta: Callable[[str], None],
    model: str | None = None,
) -> AgentResponse:
    """Like run_turn(), but delivers text as it's generated via `on_delta` instead of
    only once the full response is back. Deliberately NOT wrapped in with_retry(): by
    the time a transient error can occur, on_delta may have already been called with
    partial text that's already been printed to the user, so a silent retry would
    re-invoke on_delta from scratch and print duplicate, visually broken output.
    Letting the error propagate directly (same as any non-retried MazuAPIError) means
    the caller's existing error handling shows one clean message instead.
    """
    _, provider, model_name = _resolve_provider(model)
    return provider.run_turn_stream(messages, system, tools, model_name, on_delta)


def run_forced_tool(
    messages: list[dict],
    system: str,
    tool: dict,
    model: str | None = None,
) -> dict:
    """Force the model to call exactly `tool`, returning its parsed input dict."""
    _, provider, model_name = _resolve_provider(model)
    return with_retry(lambda: provider.run_forced_tool(messages, system, tool, model_name))


def summarize_usage(usage: dict) -> str:
    """Best-effort human-readable input/output/cached token summary, normalized across
    providers that name these fields differently. Anthropic marks cache reads
    explicitly (cache_control); OpenAI and DeepSeek cache matching prefixes
    automatically on their end and report it back under their own field names — this
    just surfaces whichever one is present so cache hits are visible regardless of
    provider, not just asserted. Field names have not been confirmed against a live
    OpenAI/DeepSeek response yet; unknown/missing fields are simply omitted.
    """
    total_in = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    total_out = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    cached = (
        usage.get("cache_read_input_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or 0
    )
    parts = [f"{total_in} in", f"{total_out} out"]
    if cached:
        parts.append(f"{cached} cached")
    return ", ".join(parts)
