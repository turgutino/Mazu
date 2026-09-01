import os
from typing import Callable

from mazu.llm.providers.anthropic_provider import AnthropicProvider
from mazu.llm.providers.deepseek_provider import DeepSeekProvider
from mazu.llm.providers.gemini_provider import GeminiProvider
from mazu.llm.providers.local_provider import LocalProvider
from mazu.llm.providers.openai_provider import OpenAIProvider
from mazu.llm.retry import with_retry
from mazu.llm.types import AgentResponse

_PROVIDERS = {
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
    "deepseek": DeepSeekProvider(),
    "gemini": GeminiProvider(),
    "local": LocalProvider(),
}

# Tie-breaker order only — used when nothing is configured and we have to guess among
# multiple keys that happen to be set. Not a requirement to have any specific one.
# Gemini appended at the end so it never changes default resolution for existing setups.
# "local" is deliberately absent -- it must always be chosen explicitly (--model
# local:<name>, MAZU_MODEL, or `mazu config set default_model`), never auto-detected,
# since a local server being reachable isn't a signal the user actually wants to use it.
_PROVIDER_PRIORITY = ["anthropic", "openai", "deepseek", "gemini"]

_PROVIDER_DEFAULT_MODELS = {
    "anthropic": "anthropic:claude-sonnet-5",
    "openai": "openai:gpt-5",
    "deepseek": "deepseek:deepseek-chat",
    "gemini": "gemini:gemini-2.0-flash",
    "local": "local:your-model-name",
}


def default_model() -> str:
    """Resolution order: explicit MAZU_MODEL env var, then `default_model` set via
    `mazu config set` (~/.mazu/config.toml), then auto-detect from whichever
    provider's API key is actually present in the environment, then a hardcoded
    Anthropic fallback (which will simply surface a clear "set ANTHROPIC_API_KEY" error
    via config.ensure_api_key if the user genuinely has no key configured at all).
    Nothing here assumes Anthropic is required — a DeepSeek-only or OpenAI-only setup
    is auto-detected and used with no extra flags needed.
    """
    if os.environ.get("MAZU_MODEL"):
        return os.environ["MAZU_MODEL"]

    from mazu.config import get_default_model

    configured = get_default_model()
    if configured:
        return configured

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


def default_effort() -> str | None:
    """Resolution order: MAZU_REASONING_EFFORT env var, then the `reasoning_effort`
    config value set via `mazu config set` / the mazu-web Config page -- None
    (Mazu sends nothing, leaving each model's own default thinking behavior
    untouched) if neither is set. Unlike default_model(), there is no
    auto-detection fallback -- there's no sensible provider-derived guess for how
    much a model should "think." A provider that doesn't support this at all
    (DeepSeek, local) just ignores whatever this resolves to -- see
    Provider.supports_reasoning_effort.
    """
    env_value = os.environ.get("MAZU_REASONING_EFFORT")
    if env_value:
        return env_value
    from mazu.config import get_reasoning_effort

    return get_reasoning_effort()


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
    effort = default_effort()
    return with_retry(lambda: provider.run_turn(messages, system, tools, model_name, effort=effort))


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
    effort = default_effort()
    return provider.run_turn_stream(messages, system, tools, model_name, on_delta, effort=effort)


def run_forced_tool(
    messages: list[dict],
    system: str,
    tool: dict,
    model: str | None = None,
) -> tuple[dict, dict]:
    """Force the model to call exactly `tool`, returning (parsed input dict, usage
    dict) -- usage is passed through, not discarded, so callers can log real cost.
    """
    _, provider, model_name = _resolve_provider(model)
    return with_retry(lambda: provider.run_forced_tool(messages, system, tool, model_name))


def list_models(provider_name: str) -> list[str]:
    """Live model catalog for `provider_name` (e.g. "deepseek"), fetched from that
    provider's own API using whichever key is currently configured -- see
    Provider.list_models() for why this is a live call, not the static
    mazu/llm/capabilities.py table. Deliberately not wrapped in with_retry(): a
    bad/missing key should surface immediately as a clear MazuAuthError, not be
    retried as if it might transiently fix itself.

    The missing-key check below is deliberately done HERE rather than left to each
    provider's own client construction: AnthropicProvider's _get_client() passes a
    missing key through to the SDK as `None` on purpose (see
    test_anthropic_provider_key_isolation.py's test_missing_key_falls_back_to_none_
    not_a_crash) so the SDK's own env fallback still has a chance to work -- real
    bug found via live testing (mazu-web's new model-picker endpoint) is that this
    means a genuinely missing key surfaces as a raw SDK TypeError deep in header
    construction (not an anthropic.AnthropicError, so run_turn's own error mapping
    never sees it either) instead of a clean, catchable error. Checking here catches
    it before any provider-specific client is ever built, for every provider.
    """
    from mazu.llm.errors import MazuAuthError

    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {', '.join(_PROVIDERS)}")
    if provider.requires_api_key and not os.environ.get(provider.api_key_env):
        raise MazuAuthError(
            f"{provider.api_key_env} is not set. Set it to list {provider_name}: models."
        )
    return provider.list_models()


def extract_cache_tokens(usage: dict) -> tuple[int, int]:
    """Returns (cache_read_tokens, cache_write_tokens), normalized across providers
    that name these fields differently -- the single place this field-name mapping
    lives, reused by both summarize_usage() (display) and every real cost-logging
    call site (mazu/agent/loop.py, mazu_web/chat_session.py, mazu/agent/compaction.py,
    mazu/agent/session.py), so a provider's exact field names never need duplicating.

    Anthropic marks cache reads/writes explicitly via `cache_read_input_tokens` /
    `cache_creation_input_tokens` (see anthropic_provider.py's cache_control usage).
    OpenAI and DeepSeek cache matching prefixes automatically server-side with no
    explicit "write" concept exposed -- confirmed live against a real DeepSeek
    response: `prompt_cache_hit_tokens` (DeepSeek's own field) and the OpenAI-
    standard `prompt_tokens_details.cached_tokens` are both populated identically,
    so either is a safe read-side source; there is no equivalent write-side field
    for these providers, hence cache_write_tokens is always 0 for them. Gemini
    similarly caches automatically (implicit prompt-prefix caching) with no
    explicit "write" concept Mazu requests or exposes -- its read-side figure is
    `cached_content_token_count` (see gemini_provider.py's _parse_gemini_response).
    Missing/absent fields (local models, or a provider version that doesn't report
    this) resolve to (0, 0) rather than raising -- never invented, only surfaced
    when the provider actually sends it.
    """
    cache_read = (
        usage.get("cache_read_input_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or usage.get("cached_content_token_count")
        or 0
    )
    cache_write = usage.get("cache_creation_input_tokens") or 0
    return cache_read, cache_write


def extract_cost_tokens(usage: dict) -> tuple[int, int, int, int]:
    """Returns (fresh_input_tokens, output_tokens, cache_read_tokens,
    cache_write_tokens) -- the ONLY correct inputs to pass to
    llm.pricing.estimate_cost(), used by every real cost-logging call site
    instead of each one re-deriving this.

    This exists because of a real, easy-to-miss shape difference between
    providers, confirmed live against a real DeepSeek response (two calls, the
    second reusing the first's prefix): Anthropic's `input_tokens` EXCLUDES
    cache_read_input_tokens/cache_creation_input_tokens -- they are separate,
    additive fields. But OpenAI-compatible's `prompt_tokens` INCLUDES the cached
    portion (`prompt_tokens == prompt_cache_hit_tokens + prompt_cache_miss_tokens`,
    verified: 1781 == 1664 + 117). Naively passing the raw total alongside
    cache_read_tokens into estimate_cost() double-counts the cached tokens for
    OpenAI-compatible providers -- once at full input rate (already inside the
    raw total) and again at the discounted cache rate on top. This function
    subtracts the cached portion back out for that shape only; Anthropic's
    already-exclusive `input_tokens` passes through unchanged.

    Gemini reuses Anthropic's field NAME (`input_tokens`, see
    gemini_provider.py's _parse_gemini_response) but not Anthropic's exclusive
    SEMANTICS -- per Gemini's own API docs, `prompt_token_count` is the total
    prompt size and `cached_content_token_count` is a subset of it, the same
    inclusive convention as OpenAI-compatible's `prompt_tokens`. Branching on the
    `input_tokens` key name alone would misclassify Gemini as Anthropic-shaped and
    double-count its cached tokens the same way the original OpenAI-compatible bug
    did. Gemini's usage dict always carries `cached_content_token_count` (0 when
    unused, never absent -- see _parse_gemini_response), which is what
    distinguishes it from a real Anthropic response here.

    Deliberately separate from the raw `usage.get("input_tokens", ...)` extraction
    still used for `usage_store.log()`'s stored `input_tokens` column and for
    summarize_usage()'s display -- those intentionally keep showing the RAW total
    (unchanged historical meaning, matches what the provider actually billed as
    "prompt size"); only the COST math needs the fresh/cached split.
    """
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    cache_read_tokens, cache_write_tokens = extract_cache_tokens(usage)
    if "cached_content_token_count" in usage:
        raw_total = usage.get("input_tokens", 0) or 0
        fresh_input_tokens = max(raw_total - cache_read_tokens, 0)
    elif "input_tokens" in usage:
        fresh_input_tokens = usage.get("input_tokens") or 0
    else:
        raw_total = usage.get("prompt_tokens", 0) or 0
        fresh_input_tokens = max(raw_total - cache_read_tokens, 0)
    return fresh_input_tokens, output_tokens, cache_read_tokens, cache_write_tokens


def summarize_usage(usage: dict) -> str:
    """Best-effort human-readable input/output/cached token summary -- see
    extract_cache_tokens() for the cross-provider field-name normalization this
    relies on for the "cached" figure.
    """
    total_in = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    total_out = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    cached, _ = extract_cache_tokens(usage)
    parts = [f"{total_in} in", f"{total_out} out"]
    if cached:
        parts.append(f"{cached} cached")
    return ", ".join(parts)
