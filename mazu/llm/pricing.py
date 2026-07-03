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
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Returns an approximate USD cost, or None if `model` has no pricing entry
    (unknown/newer model) — callers should treat None as "can't estimate", not zero.
    """
    rates = PRICING_PER_MILLION_TOKENS.get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
