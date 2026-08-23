import pytest

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


def test_estimate_cost_cache_read_tokens_are_cheaper_than_fresh_input():
    # anthropic:claude-sonnet-5 input rate is 3.0 $/1M -- a cache-read token is
    # priced at 10% of that (CACHE_READ_RATE_MULTIPLIER), i.e. 0.3 $/1M.
    base = estimate_cost("anthropic:claude-sonnet-5", 0, 0)
    with_cache_read = estimate_cost("anthropic:claude-sonnet-5", 0, 0, cache_read_tokens=1_000_000)
    assert with_cache_read - base == pytest.approx(0.3)


def test_estimate_cost_cache_write_tokens_cost_a_premium_over_fresh_input():
    # Cache writes are priced at 1.25x the base input rate (3.0 -> 3.75 $/1M).
    base = estimate_cost("anthropic:claude-sonnet-5", 0, 0)
    with_cache_write = estimate_cost("anthropic:claude-sonnet-5", 0, 0, cache_write_tokens=1_000_000)
    assert with_cache_write - base == pytest.approx(3.75)


def test_estimate_cost_with_cache_tokens_defaults_to_zero_and_matches_no_cache_call():
    # The two extra params must be purely additive: omitting them (existing call
    # sites) must produce the exact same result as passing 0 explicitly.
    omitted = estimate_cost("anthropic:claude-sonnet-5", 1000, 500)
    explicit_zero = estimate_cost(
        "anthropic:claude-sonnet-5", 1000, 500, cache_read_tokens=0, cache_write_tokens=0
    )
    assert omitted == explicit_zero


def test_estimate_cost_unknown_model_returns_none_even_with_cache_tokens():
    assert (
        estimate_cost(
            "someprovider:unknown-model", 100, 100, cache_read_tokens=1000, cache_write_tokens=1000
        )
        is None
    )
