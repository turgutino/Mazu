from pathlib import Path

import pytest

from mazu.usage.store import UsageStore


@pytest.fixture
def store(tmp_path: Path) -> UsageStore:
    s = UsageStore(tmp_path / "usage.db")
    yield s
    s.close()


def test_log_and_summary_single_call(store: UsageStore):
    store.log("chat", "s1", "anthropic", "claude-sonnet-5", 100, 50, 0.001050)

    summary = store.summary()

    assert summary["total_calls"] == 1
    assert summary["total_cost"] == pytest.approx(0.001050)
    assert summary["has_unpriced_calls"] is False
    assert len(summary["by_model"]) == 1
    assert summary["by_model"][0]["provider"] == "anthropic"
    assert summary["by_model"][0]["model"] == "claude-sonnet-5"
    assert summary["by_model"][0]["calls"] == 1
    assert summary["by_model"][0]["input_tokens"] == 100
    assert summary["by_model"][0]["output_tokens"] == 50


def test_summary_aggregates_multiple_calls_same_model(store: UsageStore):
    store.log("chat", "s1", "deepseek", "deepseek-chat", 100, 50, 0.01)
    store.log("run", "s2", "deepseek", "deepseek-chat", 200, 100, 0.02)

    summary = store.summary()

    assert summary["total_calls"] == 2
    assert summary["total_cost"] == pytest.approx(0.03)
    assert len(summary["by_model"]) == 1
    row = summary["by_model"][0]
    assert row["calls"] == 2
    assert row["input_tokens"] == 300
    assert row["output_tokens"] == 150


def test_summary_groups_by_provider_and_model_separately(store: UsageStore):
    store.log("chat", "s1", "anthropic", "claude-sonnet-5", 100, 50, 1.0)
    store.log("chat", "s2", "deepseek", "deepseek-chat", 100, 50, 0.01)

    summary = store.summary()

    assert summary["total_calls"] == 2
    assert len(summary["by_model"]) == 2
    # ORDER BY cost DESC -- the more expensive model should come first.
    assert summary["by_model"][0]["model"] == "claude-sonnet-5"
    assert summary["by_model"][1]["model"] == "deepseek-chat"


def test_summary_handles_unpriced_model_without_dropping_cost(store: UsageStore):
    store.log("chat", "s1", "anthropic", "claude-sonnet-5", 100, 50, 1.0)
    store.log("chat", "s2", "openai", "some-new-unpriced-model", 100, 50, None)

    summary = store.summary()

    assert summary["total_calls"] == 2
    assert summary["total_cost"] == pytest.approx(1.0)  # NULL contributes 0, not dropped
    assert summary["has_unpriced_calls"] is True


def test_summary_empty_store_returns_zeros_not_crash(store: UsageStore):
    summary = store.summary()

    assert summary == {
        "by_model": [],
        "total_cost": 0.0,
        "total_calls": 0,
        "has_unpriced_calls": False,
    }


def test_summary_since_days_filters_old_rows(store: UsageStore):
    # log() always stamps "now" -- insert an old row directly to test the time filter.
    store.conn.execute(
        "INSERT INTO usage_log "
        "(created_at, command, session_id, provider, model, input_tokens, output_tokens, estimated_cost_usd) "
        "VALUES ('2020-01-01T00:00:00+00:00', 'chat', 's-old', 'anthropic', 'claude-sonnet-5', 100, 50, 1.0)"
    )
    store.conn.commit()
    store.log("chat", "s-new", "anthropic", "claude-sonnet-5", 100, 50, 2.0)

    all_time = store.summary()
    recent = store.summary(since_days=7)

    assert all_time["total_calls"] == 2
    assert all_time["total_cost"] == pytest.approx(3.0)
    assert recent["total_calls"] == 1
    assert recent["total_cost"] == pytest.approx(2.0)
