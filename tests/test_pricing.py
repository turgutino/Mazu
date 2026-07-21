from mazu.llm.pricing import estimate_cost


def test_estimate_cost_known_model_is_correct():
    # anthropic:claude-sonnet-5 is (3.0, 15.0) $/1M tokens.
    cost = estimate_cost("anthropic:claude-sonnet-5", 1_000_000, 1_000_000)
    assert cost == 18.0


def test_estimate_cost_unknown_model_returns_none_not_zero():
    assert estimate_cost("someprovider:unknown-model", 1_000_000, 1_000_000) is None


def test_estimate_cost_gpt4o_family_and_gemini_are_priced():
    assert estimate_cost("openai:gpt-4o", 1_000_000, 1_000_000) == 12.5
    assert estimate_cost("openai:gpt-4o-mini", 1_000_000, 1_000_000) == 0.75
    assert estimate_cost("gemini:gemini-2.0-flash", 1_000_000, 1_000_000) == 0.5


def test_estimate_cost_zero_tokens_is_zero_not_none():
    assert estimate_cost("anthropic:claude-sonnet-5", 0, 0) == 0.0
