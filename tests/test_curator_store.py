import pytest

from mazu.curator.store import CuratorStore, new_curator_run_id


@pytest.fixture
def store(tmp_path):
    s = CuratorStore(tmp_path / "curator.db")
    yield s
    s.close()


def test_new_run_id_is_unique():
    assert new_curator_run_id() != new_curator_run_id()


def test_start_and_finish_run(store: CuratorStore):
    run_id = new_curator_run_id()
    store.start_run(run_id, "anthropic:claude-haiku-4-5", ["memory"], dry_run=False)
    run = store.last_run(run_id)
    assert run["status"] == "running"
    assert run["areas"] == "memory"

    store.finish_run(run_id, "completed", None, 0.0123)
    run = store.last_run(run_id)
    assert run["status"] == "completed"
    assert run["total_cost_usd"] == 0.0123


def test_log_entry_and_log_for_run(store: CuratorStore):
    run_id = new_curator_run_id()
    store.start_run(run_id, "anthropic:claude-haiku-4-5", ["memory"], dry_run=False)
    store.log_entry(
        run_id=run_id, area="memory", action="archive_memory", rationale="stale",
        target_type="memory", target_id="3", reversal_hint="mazu memory unarchive 3",
    )
    entries = store.log_for_run(run_id)
    assert len(entries) == 1
    assert entries[0]["action"] == "archive_memory"
    assert entries[0]["rationale"] == "stale"
    assert entries[0]["applied"] == 1


def test_log_entry_dry_run_marks_not_applied(store: CuratorStore):
    run_id = new_curator_run_id()
    store.log_entry(run_id=run_id, area="memory", action="archive_memory", rationale="x", applied=False)
    entries = store.log_for_run(run_id)
    assert entries[0]["applied"] == 0


def test_log_recent_filters_by_area(store: CuratorStore):
    run_id = new_curator_run_id()
    store.log_entry(run_id=run_id, area="memory", action="a", rationale="x")
    store.log_entry(run_id=run_id, area="skills", action="b", rationale="y")
    rows = store.log_recent(area="memory")
    assert len(rows) == 1
    assert rows[0]["area"] == "memory"


def test_watermark_starts_unset(store: CuratorStore):
    assert store.get_watermark("memory") is None


def test_advance_watermark_then_read_back(store: CuratorStore):
    run_id = new_curator_run_id()
    store.advance_watermark("memory", run_id)
    watermark = store.get_watermark("memory")
    assert watermark["last_run_id"] == run_id
    assert watermark["last_run_at"] is not None


def test_advance_watermark_twice_updates_in_place(store: CuratorStore):
    run_id_1 = new_curator_run_id()
    run_id_2 = new_curator_run_id()
    store.advance_watermark("memory", run_id_1)
    store.advance_watermark("memory", run_id_2)
    watermark = store.get_watermark("memory")
    assert watermark["last_run_id"] == run_id_2


def test_migration_is_idempotent_across_reopen(tmp_path):
    db_path = tmp_path / "curator.db"
    s1 = CuratorStore(db_path)
    s1.close()
    s2 = CuratorStore(db_path)  # re-running schema/executescript must not error
    s2.close()
