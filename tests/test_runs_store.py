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
