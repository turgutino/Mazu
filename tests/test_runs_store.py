from pathlib import Path

import pytest

from mazu.runs.store import RunStore


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    s = RunStore(tmp_path / "runs.db")
    yield s
    s.close()


def test_start_creates_a_running_row(store: RunStore):
    store.start("r1", "do something", "anthropic:claude-sonnet-5", 15, 1, False, None, None, False)
    row = store.get("r1")
    assert row is not None
    assert row["status"] == "running"
    assert row["task"] == "do something"
    assert row["model"] == "anthropic:claude-sonnet-5"
    assert row["max_steps"] == 15
    assert row["allow_shell"] == 0
    assert row["dry_run"] == 0
    assert row["last_step"] == 0
    assert row["checkpoints_created"] == 0
    assert row["ended_at"] is None


def test_start_stores_shell_allowlist_as_comma_joined_text(store: RunStore):
    store.start("r1", "x", None, 15, 1, True, ["git", "npm"], None, False)
    row = store.get("r1")
    assert row["shell_allowlist"] == "git,npm"


def test_start_stores_none_shell_allowlist_as_null(store: RunStore):
    store.start("r1", "x", None, 15, 1, True, None, None, False)
    row = store.get("r1")
    assert row["shell_allowlist"] is None


def test_get_missing_run_returns_none(store: RunStore):
    assert store.get("nope") is None


def test_update_progress_without_checkpoint_bumps_last_step_only(store: RunStore):
    store.start("r1", "x", None, 15, 1, False, None, None, False)
    store.update_progress("r1", 3)
    row = store.get("r1")
    assert row["last_step"] == 3
    assert row["checkpoints_created"] == 0
    assert row["last_checkpoint_id"] is None


def test_update_progress_with_checkpoint_increments_count_and_sets_last_id(store: RunStore):
    store.start("r1", "x", None, 15, 1, False, None, None, False)
    store.update_progress("r1", 1, checkpoint_id="cp_000001")
    store.update_progress("r1", 2, checkpoint_id="cp_000002")
    row = store.get("r1")
    assert row["last_step"] == 2
    assert row["checkpoints_created"] == 2
    assert row["last_checkpoint_id"] == "cp_000002"


def test_finish_sets_status_stop_reason_and_ended_at(store: RunStore):
    store.start("r1", "x", None, 15, 1, False, None, None, False)
    store.finish("r1", status="completed", stop_reason="end_turn", memories_saved=2)
    row = store.get("r1")
    assert row["status"] == "completed"
    assert row["stop_reason"] == "end_turn"
    assert row["ended_at"] is not None
    assert row["memories_saved"] == 2


def test_finish_accumulates_memories_saved_across_calls(store: RunStore):
    # A resumed run calls finish() again on the same row -- memories_saved should
    # accumulate across the original run and its resumption(s), not overwrite.
    store.start("r1", "x", None, 15, 1, False, None, None, False)
    store.finish("r1", status="stopped", stop_reason="max_steps", memories_saved=1)
    store.finish("r1", status="completed", stop_reason="end_turn", memories_saved=2)
    row = store.get("r1")
    assert row["memories_saved"] == 3
    assert row["status"] == "completed"
    assert row["stop_reason"] == "end_turn"


def test_list_runs_orders_most_recent_first(store: RunStore):
    store.start("older", "x", None, 15, 1, False, None, None, False)
    store.start("newer", "x", None, 15, 1, False, None, None, False)
    rows = store.list_runs()
    assert rows[0]["id"] == "newer"


def test_list_runs_respects_limit(store: RunStore):
    for i in range(5):
        store.start(f"r{i}", "x", None, 15, 1, False, None, None, False)
    assert len(store.list_runs(limit=2)) == 2


def test_list_runs_empty_store(store: RunStore):
    assert store.list_runs() == []


# ---------------------------------------------------------------------------
# explore-outcome columns (Learning Model Router addendum)
# ---------------------------------------------------------------------------


def test_new_db_already_has_explore_outcome_columns(store: RunStore):
    columns = {row["name"] for row in store.conn.execute("PRAGMA table_info(runs)")}
    assert "explore_group_id" in columns
    assert "test_passed" in columns


def test_migration_adds_explore_outcome_columns_to_a_pre_existing_db(tmp_path: Path):
    """A .mazu/runs.db created before this addendum shipped has neither column --
    opening it with the new RunStore must not crash, and must add the columns
    (same regression shape as the existing lineage-column migration test)."""
    db_path = tmp_path / "old_runs.db"
    old = RunStore(db_path)
    old.start("r1", "x", None, 15, 1, False, None, None, False)
    old.close()

    import sqlite3

    raw = sqlite3.connect(db_path)
    raw.execute("ALTER TABLE runs DROP COLUMN explore_group_id")
    raw.execute("ALTER TABLE runs DROP COLUMN test_passed")
    raw.commit()
    raw.close()

    migrated = RunStore(db_path)
    columns = {row["name"] for row in migrated.conn.execute("PRAGMA table_info(runs)")}
    assert "explore_group_id" in columns
    assert "test_passed" in columns
    row = migrated.get("r1")
    assert row["explore_group_id"] is None
    assert row["test_passed"] is None
    migrated.close()


def test_set_explore_outcome_writes_group_id_and_test_passed(store: RunStore):
    store.start("r1", "x", None, 15, 1, False, None, None, False)
    store.set_explore_outcome("r1", "abc12345", True)
    row = store.get("r1")
    assert row["explore_group_id"] == "abc12345"
    assert row["test_passed"] == 1


def test_set_explore_outcome_false_test_passed(store: RunStore):
    store.start("r1", "x", None, 15, 1, False, None, None, False)
    store.set_explore_outcome("r1", "abc12345", False)
    row = store.get("r1")
    assert row["test_passed"] == 0


def test_set_explore_outcome_null_test_passed_when_no_test_command(store: RunStore):
    store.start("r1", "x", None, 15, 1, False, None, None, False)
    store.set_explore_outcome("r1", "abc12345", None)
    row = store.get("r1")
    assert row["explore_group_id"] == "abc12345"
    assert row["test_passed"] is None


def test_ordinary_run_has_null_explore_columns_by_default(store: RunStore):
    store.start("r1", "x", None, 15, 1, False, None, None, False)
    row = store.get("r1")
    assert row["explore_group_id"] is None
    assert row["test_passed"] is None
