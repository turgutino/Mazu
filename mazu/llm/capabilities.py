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

# Maximum output tokens Anthropic will generate in a single turn, self-reported by
# each model's own documentation at the time this table was written -- same
# "best-effort, will go stale, verify against provider docs before relying on it"
# character as CONTEXT_WINDOW_TOKENS above. Anthropic's `max_tokens` request field
# is REQUIRED (unlike OpenAI-compatible/Gemini, which both accept an unbounded
# request and cap output at the model's own ceiling with no client-side value
# needed) -- so a real per-model table is what avoids Mazu imposing an artificial,
# needlessly low ceiling on a model that supports far more, while still not
# blindly sending some enormous made-up number for a model that can't honor it.
# Only Anthropic needs this table: OpenAI-compatible/Gemini providers never pass
# a max-output-tokens parameter at all today, so their models are unaffected by
# this cap (see anthropic_provider.py's _max_output_tokens; other providers'
# run_turn/run_turn_stream have no equivalent parameter).
#
# Hard ceiling caught live: mazu/agent/autonomous.py (the engine behind `mazu
# run`/`mazu explore`) calls Anthropic exclusively through the NON-streaming
# run_turn, never run_turn_stream. The Anthropic Python SDK's own client-side
# guard (_calculate_nonstreaming_timeout) rejects any non-streaming request
# whose max_tokens implies more than 10 minutes of possible generation time --
# concretely, max_tokens > 21_333 -- raising a ValueError before the request is
# even sent, with zero API cost incurred. Every value below MUST stay under
# that ceiling, or Anthropic becomes silently unusable for every `mazu run`/
# `mazu explore` invocation. (Streaming call sites like `mazu chat` have no such
# limit; this constraint is specific to the non-streaming path.)
MAX_OUTPUT_TOKENS: dict[str, int] = {
    "anthropic:claude-opus-4-8": 16_000,
    "anthropic:claude-sonnet-5": 16_000,
    "anthropic:claude-haiku-4-5": 8_192,
}
# Fallback for any Anthropic model not in the table above (an older or
# newer/unreleased-at-table-writing-time model) -- deliberately conservative
# (higher than the old flat 4096 that motivated this table, but not an arbitrary
# huge number that could fail against a real, lower-ceiling model this table
# simply doesn't know about yet).
DEFAULT_ANTHROPIC_MAX_TOKENS = 8_192

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
