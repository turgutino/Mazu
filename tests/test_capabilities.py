from mazu.llm.capabilities import (
    CONTEXT_WINDOW_TOKENS,
    REAL_STREAMING,
    TOOL_USE_SUPPORTED,
    list_capabilities,
)
from mazu.llm.client import _PROVIDER_DEFAULT_MODELS
from mazu.llm.pricing import PRICING_PER_MILLION_TOKENS


def test_list_capabilities_includes_every_priced_model():
    rows = list_capabilities()
    keys = {f"{r.provider}:{r.model}" for r in rows}
    assert set(PRICING_PER_MILLION_TOKENS).issubset(keys)


def test_list_capabilities_includes_every_provider_default_even_if_unpriced():
    # Every current provider default happens to have a pricing entry now, but the
    # inclusion itself must not depend on that -- a provider default without pricing
    # (the situation Gemini used to be in, before its entry was added) must still
    # appear in the table, per list_capabilities()'s own docstring.
    rows = list_capabilities()
    keys = {f"{r.provider}:{r.model}" for r in rows}
    assert set(_PROVIDER_DEFAULT_MODELS.values()).issubset(keys)


def test_list_capabilities_no_duplicate_keys():
    rows = list_capabilities()
    keys = [f"{r.provider}:{r.model}" for r in rows]
    assert len(keys) == len(set(keys))


def test_list_capabilities_sorted_by_key():
    rows = list_capabilities()
    keys = [f"{r.provider}:{r.model}" for r in rows]
    assert keys == sorted(keys)


def test_priced_model_has_a_real_price():
    rows = {f"{r.provider}:{r.model}": r for r in list_capabilities()}
    row = rows["anthropic:claude-sonnet-5"]
    assert row.input_price_per_million == 3.0
    assert row.output_price_per_million == 15.0


def test_unpriced_model_reports_none_price_not_zero(monkeypatch):
    # Every current provider default happens to be priced now (gpt-4o/gpt-4o-mini/
    # gemini-2.0-flash were added to close that gap) -- simulate a provider whose
    # default model genuinely has no pricing entry, the situation Gemini used to be
    # in, to confirm list_capabilities() still reports None rather than silently
    # defaulting to 0 (which would make an unpriced model look free instead of
    # "can't estimate").
    import mazu.llm.client as client_module

    # list_capabilities() imports _PROVIDER_DEFAULT_MODELS from mazu.llm.client
    # locally, inside the function body, so the patch target is the client module's
    # own attribute, not a re-export on capabilities.py.
    monkeypatch.setattr(
        client_module, "_PROVIDER_DEFAULT_MODELS", {"fakeprovider": "fakeprovider:fake-model"}
    )
    rows = {f"{r.provider}:{r.model}": r for r in list_capabilities()}
    row = rows["fakeprovider:fake-model"]
    assert row.input_price_per_million is None
    assert row.output_price_per_million is None


def test_streaming_and_tool_use_reflect_the_provider_tables():
    rows = {f"{r.provider}:{r.model}": r for r in list_capabilities()}
    row = rows["anthropic:claude-sonnet-5"]
    assert row.streaming == REAL_STREAMING["anthropic"]
    assert row.tool_use == TOOL_USE_SUPPORTED["anthropic"]

    gemini_default = _PROVIDER_DEFAULT_MODELS["gemini"]
    gemini_row = rows[gemini_default]
    assert gemini_row.streaming is False  # documented as deferred, not yet verified
    assert gemini_row.tool_use is True


def test_context_window_present_when_listed():
    rows = {f"{r.provider}:{r.model}": r for r in list_capabilities()}
    for key, expected_window in CONTEXT_WINDOW_TOKENS.items():
        if key in rows:
            assert rows[key].context_window == expected_window


def test_all_four_providers_appear_in_capability_table():
    rows = list_capabilities()
    providers = {r.provider for r in rows}
    assert providers == {"anthropic", "openai", "deepseek", "gemini"}
