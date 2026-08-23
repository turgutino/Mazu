"""Tests for extract_cache_tokens()/summarize_usage() -- the cross-provider
cache-token field-name normalization used both for display (summarize_usage)
and for real cost accounting (every usage_store.log() call site). Field names
verified live against a real Anthropic response shape and a real DeepSeek
response (see mazu/llm/client.py's extract_cache_tokens docstring).
"""

import pytest

from mazu.llm.client import extract_cache_tokens, extract_cost_tokens, summarize_usage


def test_extract_cache_tokens_anthropic_shape():
    usage = {
        "input_tokens": 50,
        "output_tokens": 20,
        "cache_read_input_tokens": 1792,
        "cache_creation_input_tokens": 300,
    }
    assert extract_cache_tokens(usage) == (1792, 300)


def test_extract_cache_tokens_deepseek_shape():
    # Real shape confirmed live: DeepSeek reports its own prompt_cache_hit_tokens
    # AND the OpenAI-standard prompt_tokens_details.cached_tokens, both populated.
    usage = {
        "prompt_tokens": 1882,
        "completion_tokens": 12,
        "prompt_cache_hit_tokens": 1792,
        "prompt_cache_miss_tokens": 90,
        "prompt_tokens_details": {"cached_tokens": 1792, "audio_tokens": None},
    }
    assert extract_cache_tokens(usage) == (1792, 0)  # no write-side concept for DeepSeek


def test_extract_cache_tokens_openai_standard_shape_only():
    usage = {
        "prompt_tokens": 500,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 128, "audio_tokens": None},
    }
    assert extract_cache_tokens(usage) == (128, 0)


def test_extract_cache_tokens_gemini_shape():
    # Gemini's own field, always present (0 when unused) per
    # gemini_provider.py's _parse_gemini_response -- no write-side concept,
    # since Mazu requests no explicit Gemini caching.
    usage = {"input_tokens": 1000, "output_tokens": 50, "cached_content_token_count": 800}
    assert extract_cache_tokens(usage) == (800, 0)


def test_extract_cache_tokens_gemini_shape_zero_when_unused():
    usage = {"input_tokens": 1000, "output_tokens": 50, "cached_content_token_count": 0}
    assert extract_cache_tokens(usage) == (0, 0)


def test_extract_cache_tokens_absent_fields_are_zero_not_invented():
    # Local models, or any provider version that doesn't report caching at all.
    usage = {"input_tokens": 10, "output_tokens": 5}
    assert extract_cache_tokens(usage) == (0, 0)


def test_extract_cache_tokens_empty_usage_dict():
    assert extract_cache_tokens({}) == (0, 0)


def test_summarize_usage_shows_cached_count_when_present():
    usage = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 80}
    assert summarize_usage(usage) == "100 in, 20 out, 80 cached"


def test_summarize_usage_omits_cached_when_absent():
    usage = {"input_tokens": 100, "output_tokens": 20}
    assert summarize_usage(usage) == "100 in, 20 out"


# ---------------------------------------------------------------------------
# extract_cost_tokens -- the fresh/cached split used ONLY for cost math, distinct
# from the raw totals used for display/logging. Real bug caught before shipping:
# Anthropic's input_tokens EXCLUDES cache tokens (separate fields) but OpenAI-
# compatible's prompt_tokens INCLUDES them (prompt_tokens == hit + miss, confirmed
# live against a real DeepSeek response) -- naively adding cache_read_tokens'
# discounted cost on top of the raw total double-counts those tokens for
# OpenAI-compatible providers specifically.
# ---------------------------------------------------------------------------


def test_extract_cost_tokens_anthropic_shape_is_already_exclusive():
    # Anthropic's input_tokens already excludes cache_read/cache_creation -- must
    # pass through unchanged, not have cache_read subtracted a second time.
    usage = {
        "input_tokens": 50,
        "output_tokens": 20,
        "cache_read_input_tokens": 1792,
        "cache_creation_input_tokens": 300,
    }
    fresh_input, output, cache_read, cache_write = extract_cost_tokens(usage)
    assert fresh_input == 50
    assert output == 20
    assert cache_read == 1792
    assert cache_write == 300


def test_extract_cost_tokens_gemini_shape_subtracts_cache_read_from_the_total():
    # Gemini reuses Anthropic's "input_tokens" field NAME but not its exclusive
    # semantics -- per Gemini's own API docs, prompt_token_count (here 1000) is
    # the RAW total, and cached_content_token_count (800) is a SUBSET of it, the
    # same inclusive convention as OpenAI-compatible's prompt_tokens. The fresh
    # (cost-relevant) input must be just the uncached portion (200), not the raw
    # total (1000) -- otherwise the 800 cached tokens get charged once at full
    # rate (inside the raw total) AND again at the discounted cache rate.
    usage = {"input_tokens": 1000, "output_tokens": 50, "cached_content_token_count": 800}
    fresh_input, output, cache_read, cache_write = extract_cost_tokens(usage)
    assert fresh_input == 200
    assert output == 50
    assert cache_read == 800
    assert cache_write == 0


def test_extract_cost_tokens_anthropic_shape_unaffected_by_gemini_fix():
    # A real Anthropic response never carries cached_content_token_count -- must
    # keep taking the "already exclusive, pass through unchanged" path, not be
    # misclassified as Gemini-shaped just because both use "input_tokens".
    usage = {
        "input_tokens": 50,
        "output_tokens": 20,
        "cache_read_input_tokens": 1792,
        "cache_creation_input_tokens": 300,
    }
    fresh_input, output, cache_read, cache_write = extract_cost_tokens(usage)
    assert fresh_input == 50
    assert cache_read == 1792
    assert cache_write == 300


def test_extract_cost_tokens_deepseek_shape_subtracts_cache_read_from_the_total():
    # Real shape confirmed live: prompt_tokens (1781) == prompt_cache_hit_tokens
    # (1664) + prompt_cache_miss_tokens (117). The fresh (cost-relevant) input must
    # be just the miss portion (117), not the raw total (1781) -- otherwise the
    # 1664 cached tokens get charged once at full rate (inside the raw total) AND
    # again at the discounted cache rate.
    usage = {
        "prompt_tokens": 1781,
        "completion_tokens": 47,
        "prompt_cache_hit_tokens": 1664,
        "prompt_cache_miss_tokens": 117,
        "prompt_tokens_details": {"cached_tokens": 1664, "audio_tokens": None},
    }
    fresh_input, output, cache_read, cache_write = extract_cost_tokens(usage)
    assert fresh_input == 117
    assert output == 47
    assert cache_read == 1664
    assert cache_write == 0


def test_extract_cost_tokens_openai_standard_shape_subtracts_cached_tokens():
    usage = {
        "prompt_tokens": 500,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 128, "audio_tokens": None},
    }
    fresh_input, output, cache_read, cache_write = extract_cost_tokens(usage)
    assert fresh_input == 372  # 500 - 128
    assert cache_read == 128


def test_extract_cost_tokens_no_cache_data_leaves_totals_unchanged():
    usage = {"prompt_tokens": 500, "completion_tokens": 10}
    fresh_input, output, cache_read, cache_write = extract_cost_tokens(usage)
    assert fresh_input == 500
    assert cache_read == 0
    assert cache_write == 0


def test_extract_cost_tokens_never_returns_a_negative_fresh_input():
    # Defensive: if a provider ever reports cache_read > prompt_tokens (shouldn't
    # happen, but never trust a third-party API to be perfectly self-consistent),
    # clamp at 0 rather than a nonsensical negative cost.
    usage = {"prompt_tokens": 50, "prompt_cache_hit_tokens": 999}
    fresh_input, _, cache_read, _ = extract_cost_tokens(usage)
    assert fresh_input == 0
    assert cache_read == 999


def test_extract_cost_tokens_gemini_never_returns_a_negative_fresh_input():
    # Same defensive clamp, for Gemini's inclusive shape specifically.
    usage = {"input_tokens": 50, "cached_content_token_count": 999}
    fresh_input, _, cache_read, _ = extract_cost_tokens(usage)
    assert fresh_input == 0
    assert cache_read == 999


def test_estimate_cost_with_extract_cost_tokens_does_not_double_count_deepseek():
    # The end-to-end proof: computing cost from extract_cost_tokens()'s output
    # must equal (fresh input at full rate) + (cache read at discount) + output --
    # never (raw total at full rate) + (cache read at discount), which would be
    # the double-counted bug.
    from mazu.llm.pricing import estimate_cost

    usage = {
        "prompt_tokens": 1781,
        "completion_tokens": 47,
        "prompt_cache_hit_tokens": 1664,
        "prompt_cache_miss_tokens": 117,
    }
    fresh_input, output, cache_read, cache_write = extract_cost_tokens(usage)
    cost = estimate_cost(
        "deepseek:deepseek-chat", fresh_input, output,
        cache_read_tokens=cache_read, cache_write_tokens=cache_write,
    )
    # deepseek:deepseek-chat is (0.27, 1.1) $/1M; cache read at 10% of input rate.
    expected = (117 / 1_000_000) * 0.27 + (1664 / 1_000_000) * 0.27 * 0.1 + (47 / 1_000_000) * 1.1
    assert cost == pytest.approx(expected)

    # The buggy (double-counted) alternative would be noticeably higher -- assert
    # this correct result is NOT that.
    buggy_cost = (
        (1781 / 1_000_000) * 0.27 + (1664 / 1_000_000) * 0.27 * 0.1 + (47 / 1_000_000) * 1.1
    )
    assert cost < buggy_cost


def test_estimate_cost_with_extract_cost_tokens_does_not_double_count_gemini():
    # Same end-to-end proof as the DeepSeek case above, for Gemini's
    # input_tokens-but-inclusive shape.
    from mazu.llm.pricing import estimate_cost

    usage = {"input_tokens": 1000, "output_tokens": 50, "cached_content_token_count": 800}
    fresh_input, output, cache_read, cache_write = extract_cost_tokens(usage)
    cost = estimate_cost(
        "gemini:gemini-2.0-flash", fresh_input, output,
        cache_read_tokens=cache_read, cache_write_tokens=cache_write,
    )
    # gemini:gemini-2.0-flash is (0.10, 0.40) $/1M; cache read at 10% of input rate.
    expected = (200 / 1_000_000) * 0.10 + (800 / 1_000_000) * 0.10 * 0.1 + (50 / 1_000_000) * 0.40
    assert cost == pytest.approx(expected)

    buggy_cost = (
        (1000 / 1_000_000) * 0.10 + (800 / 1_000_000) * 0.10 * 0.1 + (50 / 1_000_000) * 0.40
    )
    assert cost < buggy_cost
