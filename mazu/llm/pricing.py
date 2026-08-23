"""Best-effort, approximate per-model pricing, used only as a safety-net spend
estimate for `mazu run --max-cost`. This table WILL go stale as providers change
prices or ship new models — it is not a billing system, just a rough backstop to
stop a runaway autonomous run before it burns real money. Rates are USD per
million tokens. A model not listed here simply can't have its cost estimated;
`--max-cost` is then ignored with a one-time warning rather than blocking the run.
"""

# (input $/1M tokens, output $/1M tokens)
PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "anthropic:claude-opus-4-8": (15.0, 75.0),
    "anthropic:claude-sonnet-5": (3.0, 15.0),
    "anthropic:claude-haiku-4-5": (0.8, 4.0),
    "openai:gpt-5": (5.0, 15.0),
    "openai:gpt-5-mini": (0.5, 2.0),
    "deepseek:deepseek-chat": (0.27, 1.1),
    "deepseek:deepseek-reasoner": (0.55, 2.19),
    # best-effort — verify against provider docs before relying on these for real budgeting
    "openai:gpt-4o": (2.50, 10.0),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "gemini:gemini-2.0-flash": (0.10, 0.40),
}


# Cached tokens are priced as a multiple of the model's own (uncached) input rate,
# not a separate per-model table -- these ratios are what Anthropic publishes for
# prompt caching (cache read ~= 1/10th of a fresh input token; a cache write costs
# ~1.25x a fresh input token, for the standard 5-minute-TTL ephemeral cache) and are
# used as a best-effort approximation for every provider, matching this file's
# existing "rough backstop, not a billing system" character. DeepSeek/OpenAI don't
# publish an explicit cache-write price (their caching is automatic and reported
# read-only, see mazu/llm/client.py's extract_cache_tokens) so cache_write_tokens is
# simply 0 for those providers' usage dicts -- the multiplier is never applied there.
CACHE_READ_RATE_MULTIPLIER = 0.1
CACHE_WRITE_RATE_MULTIPLIER = 1.25


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Returns an approximate USD cost, or None if `model` has no pricing entry
    (unknown/newer model) — callers should treat None as "can't estimate", not zero.

    `cache_read_tokens`/`cache_write_tokens` default to 0, so every existing call
    site (which never passes them) computes byte-for-byte the same cost as before --
    this is a pure additive extension, not a behavior change for anyone not passing
    the new arguments.
    """
    rates = PRICING_PER_MILLION_TOKENS.get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
    if cache_read_tokens:
        cost += (cache_read_tokens / 1_000_000) * input_rate * CACHE_READ_RATE_MULTIPLIER
    if cache_write_tokens:
        cost += (cache_write_tokens / 1_000_000) * input_rate * CACHE_WRITE_RATE_MULTIPLIER
    return cost
