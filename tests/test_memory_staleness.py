from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mazu.memory.staleness import apply_archival, find_stale_candidates
from mazu.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


def _backdate(store: MemoryStore, memory_id: int, days_ago: int, also_last_used: bool = False) -> None:
    """Test-only helper: store.add() always timestamps with "now", so staleness
    (which depends entirely on created_at/last_used_at being old) can only be
    exercised by reaching into the DB directly, same technique this project's own
    other tests use to simulate pre-existing state add() has no API for.
    """
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    if also_last_used:
        store.conn.execute("UPDATE memories SET created_at = ?, last_used_at = ? WHERE id = ?", (ts, ts, memory_id))
    else:
        store.conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (ts, memory_id))
    store.conn.commit()


def test_no_candidates_when_nothing_is_old_enough(store: MemoryStore):
    store.add(category="fact", title="fresh fact", body="just added")
    assert find_stale_candidates(store, min_days=30) == []


def test_flags_an_old_memory_never_retrieved(store: MemoryStore):
    memory_id = store.add(category="fact", title="old fact", body="never used since")
    _backdate(store, memory_id, days_ago=40)

    candidates = find_stale_candidates(store, min_days=30)

    assert [c["id"] for c in candidates] == [memory_id]


def test_does_not_flag_an_old_memory_that_was_recently_used(store: MemoryStore):
    memory_id = store.add(category="fact", title="old but still relevant", body="body")
    _backdate(store, memory_id, days_ago=40)
    # Simulate a real retrieval refreshing last_used_at recently, same as
    # build_context_block()'s mark_retrieved() would.
    store.mark_retrieved([memory_id])

    assert find_stale_candidates(store, min_days=30) == []


def test_young_memory_never_flagged_even_with_no_retrieval_yet(store: MemoryStore):
    # Created 5 days ago, never retrieved -- but hasn't had a fair 30-day chance
    # yet, so this must NOT be treated the same as genuine staleness.
    memory_id = store.add(category="fact", title="brand new", body="body")
    _backdate(store, memory_id, days_ago=5)

    assert find_stale_candidates(store, min_days=30) == []


def test_pinned_memory_never_flagged_regardless_of_age(store: MemoryStore):
    memory_id = store.add(category="fact", title="old but pinned", body="body", pinned=True)
    _backdate(store, memory_id, days_ago=100)

    assert find_stale_candidates(store, min_days=30) == []


def test_recent_mistake_floor_excluded_but_older_mistakes_are_eligible(store: MemoryStore):
    # build_context_block's floor always includes the 3 *most recent* mistakes
    # regardless of relevance -- their last_used_at is refreshed every session no
    # matter what, so staleness can't apply to them. A 4th, older mistake beyond
    # that floor is NOT protected this way and should be a real candidate.
    older_id = store.add(category="mistake", title="oldest mistake", body="body")
    recent_ids = [store.add(category="mistake", title=f"mistake {i}", body="body") for i in range(3)]

    for mid in [older_id, *recent_ids]:
        _backdate(store, mid, days_ago=40)

    candidates = {c["id"] for c in find_stale_candidates(store, min_days=30)}

    assert older_id in candidates
    for rid in recent_ids:
        assert rid not in candidates


def test_superseded_and_archived_memories_are_not_candidates(store: MemoryStore):
    old_id = store.add(category="fact", title="old", body="body")
    new_id = store.add(category="fact", title="new", body="body")
    store.supersede(old_id, new_id)
    _backdate(store, old_id, days_ago=40)

    already_archived_id = store.add(category="fact", title="already archived", body="body")
    store.archive(already_archived_id)
    _backdate(store, already_archived_id, days_ago=40)

    candidates = {c["id"] for c in find_stale_candidates(store, min_days=30)}
    assert old_id not in candidates
    assert already_archived_id not in candidates


def test_apply_archival_is_reversible(store: MemoryStore):
    memory_id = store.add(category="fact", title="stale fact", body="body")
    _backdate(store, memory_id, days_ago=40)

    candidates = find_stale_candidates(store, min_days=30)
    summary = apply_archival(store, candidates)

    assert summary == [{"id": memory_id, "category": "fact", "title": "stale fact"}]
    assert memory_id not in {r["id"] for r in store.all_active()}

    store.unarchive(memory_id)
    assert memory_id in {r["id"] for r in store.all_active()}


def test_apply_archival_on_empty_candidates_does_nothing(store: MemoryStore):
    store.add(category="fact", title="fresh", body="body")
    assert apply_archival(store, []) == []
    assert len(store.all_active()) == 1
