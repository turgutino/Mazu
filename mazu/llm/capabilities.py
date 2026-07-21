"""Best-effort model capability table -- context window size, whether Mazu has real
(token-by-token) streaming implemented for that provider, tool-use support, and
approximate pricing (reused from pricing.py) -- surfaced via `mazu models` for anyone
choosing between providers/models. Like pricing.py, this WILL go stale as providers
ship new models or change limits; entries not listed here simply show as unknown
rather than a guessed number.
"""

from dataclasses import dataclass

from mazu.llm.pricing import PRICING_PER_MILLION_TOKENS

# Context window sizes in tokens, self-reported by each provider's own documentation
# at the time this table was written -- NOT independently verified against a live API
# call (there is no endpoint that returns this), and it will go stale as providers
# change limits. A model not listed here shows as unknown rather than a guessed number.
CONTEXT_WINDOW_TOKENS: dict[str, int] = {
    "anthropic:claude-opus-4-8": 200_000,
    "anthropic:claude-sonnet-5": 200_000,
    "anthropic:claude-haiku-4-5": 200_000,
    "openai:gpt-5": 400_000,
    "openai:gpt-5-mini": 400_000,
    "deepseek:deepseek-chat": 128_000,
    "deepseek:deepseek-reasoner": 128_000,
    "gemini:gemini-2.0-flash": 1_000_000,
}

# Whether Mazu implements REAL token-by-token streaming for this provider (vs.
# falling back to Provider's default run_turn_stream, which waits for the complete
# response and then delivers it to on_delta as a single chunk -- still correct output,
# just not incremental). Keyed by provider name, not model: streaming support is a
# per-provider implementation detail in mazu/llm/providers/, not a per-model one.
REAL_STREAMING: dict[str, bool] = {
    "anthropic": True,
    "openai": True,
    "deepseek": True,
    "gemini": False,  # see GeminiProvider's docstring: chunk-level function-call
    # behavior isn't verifiable from static SDK inspection alone; deferred until it
    # can be checked against the live API.
    "local": True,  # same OpenAICompatibleProvider code path as openai/deepseek
}

# All four current providers support tool/function calling -- Mazu's whole agent loop
# depends on it, so a provider without it wouldn't be usable at all. Kept as an
# explicit table (not just an assumed constant True) so a future tool-incapable
# model/provider has somewhere to say so, instead of the table silently lying.
TOOL_USE_SUPPORTED: dict[str, bool] = {
    "anthropic": True,
    "openai": True,
    "deepseek": True,
    "gemini": True,
    "local": True,
}


@dataclass
class ModelCapability:
    provider: str
    model: str
    streaming: bool
    tool_use: bool
    context_window: int | None
    input_price_per_million: float | None
    output_price_per_million: float | None


def list_capabilities() -> list["ModelCapability"]:
    """One row per model Mazu knows about: every model with a pricing.py entry, plus
    each provider's own default model (so a provider with no priced model, like
    Gemini today, still shows up). Deduped and sorted by "provider:model" key.
    """
    from mazu.llm.client import _PROVIDER_DEFAULT_MODELS

    model_keys = set(PRICING_PER_MILLION_TOKENS) | set(_PROVIDER_DEFAULT_MODELS.values())
    rows = []
    for key in sorted(model_keys):
        provider, model = key.split(":", 1)
        rates = PRICING_PER_MILLION_TOKENS.get(key)
        rows.append(
            ModelCapability(
                provider=provider,
                model=model,
                streaming=REAL_STREAMING.get(provider, False),
                tool_use=TOOL_USE_SUPPORTED.get(provider, False),
                context_window=CONTEXT_WINDOW_TOKENS.get(key),
                input_price_per_million=rates[0] if rates else None,
                output_price_per_million=rates[1] if rates else None,
            )
        )
    return rows
