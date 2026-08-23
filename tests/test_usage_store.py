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
        "total_cache_read_tokens": 0,
        "total_cache_write_tokens": 0,
    }


# ---------------------------------------------------------------------------
# cache_read_tokens / cache_write_tokens (additive migration)
# ---------------------------------------------------------------------------


def test_log_without_cache_tokens_stores_null_not_zero(store: UsageStore):
    store.log("chat", "s1", "anthropic", "claude-sonnet-5", 100, 50, 0.001)

    row = store.conn.execute("SELECT * FROM usage_log").fetchone()
    assert row["cache_read_tokens"] is None
    assert row["cache_write_tokens"] is None


def test_log_with_cache_tokens_persists_them(store: UsageStore):
    store.log(
        "chat", "s1", "anthropic", "claude-sonnet-5", 100, 50, 0.001,
        cache_read_tokens=1792, cache_write_tokens=300,
    )

    row = store.conn.execute("SELECT * FROM usage_log").fetchone()
    assert row["cache_read_tokens"] == 1792
    assert row["cache_write_tokens"] == 300


def test_summary_aggregates_cache_tokens_across_calls(store: UsageStore):
    store.log(
        "chat", "s1", "anthropic", "claude-sonnet-5", 100, 50, 0.001,
        cache_read_tokens=1000, cache_write_tokens=200,
    )
    store.log(
        "chat", "s1", "anthropic", "claude-sonnet-5", 100, 50, 0.001,
        cache_read_tokens=500, cache_write_tokens=0,
    )
    # A call from before this migration / a provider that never reports it -- NULL,
    # must not turn the whole SUM() into NULL/crash.
    store.log("chat", "s1", "deepseek", "deepseek-chat", 100, 50, 0.001)

    summary = store.summary()
    assert summary["total_cache_read_tokens"] == 1500
    assert summary["total_cache_write_tokens"] == 200


def test_migration_adds_cache_columns_to_a_pre_existing_database(tmp_path):
    # Simulates a real user's ~/.mazu/usage.db created before this migration --
    # the OLD schema (no cache columns), opened by the NEW UsageStore, must not
    # crash and must end up with the columns present.
    import sqlite3

    db_path = tmp_path / "old_usage.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE usage_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
        "command TEXT NOT NULL, session_id TEXT, provider TEXT NOT NULL, "
        "model TEXT NOT NULL, input_tokens INTEGER NOT NULL, "
        "output_tokens INTEGER NOT NULL, estimated_cost_usd REAL);"
    )
    conn.execute(
        "INSERT INTO usage_log (created_at, command, session_id, provider, model, "
        "input_tokens, output_tokens, estimated_cost_usd) "
        "VALUES ('2020-01-01T00:00:00', 'chat', 's1', 'anthropic', 'claude-sonnet-5', 10, 5, 0.01)"
    )
    conn.commit()
    conn.close()

    store = UsageStore(db_path)
    try:
        row = store.conn.execute("SELECT * FROM usage_log").fetchone()
        assert row["cache_read_tokens"] is None  # pre-migration row, untouched
        assert row["cache_write_tokens"] is None

        store.log(
            "chat", "s2", "anthropic", "claude-sonnet-5", 20, 10, 0.02,
            cache_read_tokens=100, cache_write_tokens=50,
        )
        summary = store.summary()
        assert summary["total_calls"] == 2
        assert summary["total_cache_read_tokens"] == 100
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Concurrent initialization (regression test for the real race caught live in
# `mazu explore`: multiple branches construct a UsageStore against the same
# brand-new db file in parallel threads, and without coordination two callers
# can both see the cache columns missing and both run ALTER TABLE -- the second
# loses with "duplicate column name: cache_read_tokens").
# ---------------------------------------------------------------------------


def test_concurrent_initialization_of_a_fresh_database_all_succeed(tmp_path):
    # sqlite3 connections can't be used/closed from a different thread than the
    # one that created them, so each thread creates its own UsageStore, logs one
    # row through it (proving the schema it sees is genuinely usable, not just
    # exception-free), and closes it itself -- all within the same thread.
    import threading

    db_path = tmp_path / "fresh_usage.db"
    thread_count = 8
    errors: list[Exception] = []
    barrier = threading.Barrier(thread_count)

    def _init_use_and_close(i: int):
        try:
            barrier.wait()  # maximize the odds every thread hits __init__ at once
            store = UsageStore(db_path)
            try:
                store.log("chat", f"s{i}", "anthropic", "claude-sonnet-5", 10, 5, 0.01)
            finally:
                store.close()
        except Exception as e:  # noqa: BLE001 -- capture any failure, including the race
            errors.append(e)

    threads = [threading.Thread(target=_init_use_and_close, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == [], f"concurrent UsageStore() init raised: {errors}"

    # Reopen fresh in the main thread to confirm the schema converged correctly
    # and every thread's row actually landed -- not just that no thread crashed.
    verify_store = UsageStore(db_path)
    try:
        summary = verify_store.summary()
        assert summary["total_calls"] == thread_count
    finally:
        verify_store.close()


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
