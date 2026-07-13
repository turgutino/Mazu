from pathlib import Path

import pytest

from mazu.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


def test_add_and_search(store: MemoryStore):
    memory_id = store.add(
        category="decision", title="Use PostgreSQL", body="Not SQLite, for concurrency."
    )
    assert memory_id > 0

    rows = store.search()
    assert len(rows) == 1
    assert rows[0]["title"] == "Use PostgreSQL"
    assert rows[0]["category"] == "decision"


def test_search_filters_by_category(store: MemoryStore):
    store.add(category="decision", title="A", body="a")
    store.add(category="mistake", title="B", body="b")

    rows = store.search(category="mistake")
    assert len(rows) == 1
    assert rows[0]["title"] == "B"


def test_search_filters_by_query_text(store: MemoryStore):
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency")
    store.add(category="decision", title="Use Redis", body="For caching")

    rows = store.search(query="Redis")
    assert len(rows) == 1
    assert rows[0]["title"] == "Use Redis"


def test_find_duplicate_exact_title_match(store: MemoryStore):
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency")

    dup = store.find_duplicate("decision", "use postgresql")  # case/whitespace-insensitive
    assert dup is not None
    assert dup["title"] == "Use PostgreSQL"


def test_find_duplicate_different_category_not_matched(store: MemoryStore):
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency")

    dup = store.find_duplicate("mistake", "Use PostgreSQL")
    assert dup is None


def test_find_duplicate_fuzzy_rephrasing_caught(store: MemoryStore):
    store.add(
        category="decision",
        title="Use PostgreSQL for the database",
        body="Chosen for concurrency and JSON support",
    )

    # Same facts, different title wording -- exact match won't catch this, fuzzy should.
    dup = store.find_duplicate(
        "decision",
        "Database is PostgreSQL",
        body="Chosen for concurrency and JSON support",
    )
    assert dup is not None


def test_find_duplicate_unrelated_fact_not_flagged(store: MemoryStore):
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency")

    dup = store.find_duplicate(
        "decision", "Use React for frontend", body="For component reuse"
    )
    assert dup is None


def test_supersede_retires_old_memory(store: MemoryStore):
    old_id = store.add(category="decision", title="Use MySQL", body="Initial choice")
    new_id = store.add(category="decision", title="Use PostgreSQL", body="Better fit")

    ok = store.supersede(old_id, new_id)
    assert ok is True

    active = store.all_active()
    active_ids = {row["id"] for row in active}
    assert old_id not in active_ids
    assert new_id in active_ids


def test_pinned_memories_returned(store: MemoryStore):
    store.add(category="fact", title="Not pinned", body="x")
    store.add(category="fact", title="Pinned", body="y", pinned=True)

    pinned = store.pinned()
    assert len(pinned) == 1
    assert pinned[0]["title"] == "Pinned"


def test_forget_deletes_memory(store: MemoryStore):
    memory_id = store.add(category="fact", title="Temp", body="x")
    assert store.forget(memory_id) is True
    assert store.search() == []


def test_forget_missing_id_returns_false(store: MemoryStore):
    assert store.forget(9999) is False


def test_recent_by_category_orders_newest_first(store: MemoryStore):
    store.add(category="mistake", title="First", body="x")
    store.add(category="mistake", title="Second", body="y")

    recent = store.recent_by_category("mistake", limit=3)
    assert [r["title"] for r in recent] == ["Second", "First"]


def test_all_active_excludes_superseded(store: MemoryStore):
    old_id = store.add(category="decision", title="Old", body="x")
    new_id = store.add(category="decision", title="New", body="y")
    store.supersede(old_id, new_id)

    active_titles = {row["title"] for row in store.all_active()}
    assert active_titles == {"New"}


# ---------------------------------------------------------------------------
# embedding storage + migration
# ---------------------------------------------------------------------------


def test_add_with_embedding_roundtrips_through_search(store: MemoryStore):
    from mazu.memory.embeddings import deserialize_embedding

    vector = [0.1, -0.2, 0.3]
    store.add(category="fact", title="Test", body="x", embedding=vector)

    row = store.search()[0]
    assert deserialize_embedding(row["embedding"]) == vector


def test_add_without_embedding_leaves_column_null(store: MemoryStore):
    store.add(category="fact", title="Test", body="x")
    row = store.search()[0]
    assert row["embedding"] is None


def test_migration_adds_embedding_column_to_pre_existing_db(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "old_memory.db"
    # Simulate a database created before the `embedding` column existed: run only
    # the pre-migration schema (everything except the new column) directly.
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            category        TEXT NOT NULL,
            title           TEXT NOT NULL,
            body            TEXT NOT NULL,
            tags            TEXT,
            source          TEXT NOT NULL,
            session_id      TEXT,
            relevance_score REAL NOT NULL DEFAULT 1.0,
            superseded_by   INTEGER REFERENCES memories(id),
            pinned          INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO memories (created_at, updated_at, category, title, body, source)
        VALUES ('2024-01-01', '2024-01-01', 'fact', 'Pre-existing memory', 'body text', 'explicit');
        """
    )
    conn.commit()
    conn.close()

    # Opening with MemoryStore must migrate the column in without losing the
    # pre-existing row.
    store = MemoryStore(db_path)
    rows = store.all_active()
    assert len(rows) == 1
    assert rows[0]["title"] == "Pre-existing memory"
    assert rows[0]["embedding"] is None  # column exists now, just empty for old rows
    store.close()


def test_migration_adds_retrieval_columns_to_pre_existing_db(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "old_memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            category        TEXT NOT NULL,
            title           TEXT NOT NULL,
            body            TEXT NOT NULL,
            tags            TEXT,
            source          TEXT NOT NULL,
            session_id      TEXT,
            relevance_score REAL NOT NULL DEFAULT 1.0,
            superseded_by   INTEGER REFERENCES memories(id),
            pinned          INTEGER NOT NULL DEFAULT 0,
            embedding       TEXT
        );
        INSERT INTO memories (created_at, updated_at, category, title, body, source)
        VALUES ('2024-01-01', '2024-01-01', 'fact', 'Pre-existing memory', 'body text', 'explicit');
        """
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path)
    rows = store.all_active()
    assert len(rows) == 1
    assert rows[0]["retrieval_count"] == 0
    assert rows[0]["last_used_at"] is None
    store.close()


# ---------------------------------------------------------------------------
# pin / unpin / edit / get
# ---------------------------------------------------------------------------


def test_pin_sets_pinned_flag(store: MemoryStore):
    memory_id = store.add(category="fact", title="A", body="a")
    assert store.pin(memory_id) is True
    assert store.get(memory_id)["pinned"] == 1


def test_unpin_clears_pinned_flag(store: MemoryStore):
    memory_id = store.add(category="fact", title="A", body="a", pinned=True)
    assert store.unpin(memory_id) is True
    assert store.get(memory_id)["pinned"] == 0


def test_pin_missing_id_returns_false(store: MemoryStore):
    assert store.pin(9999) is False


def test_get_missing_id_returns_none(store: MemoryStore):
    assert store.get(9999) is None


def test_edit_updates_title_and_body(store: MemoryStore):
    memory_id = store.add(category="fact", title="Old title", body="Old body")
    ok = store.edit(memory_id, title="New title", body="New body")
    assert ok is True
    row = store.get(memory_id)
    assert row["title"] == "New title"
    assert row["body"] == "New body"


def test_edit_partial_update_leaves_other_field_unchanged(store: MemoryStore):
    memory_id = store.add(category="fact", title="Old title", body="Old body")
    store.edit(memory_id, title="New title")
    row = store.get(memory_id)
    assert row["title"] == "New title"
    assert row["body"] == "Old body"


def test_edit_with_no_fields_returns_false_and_changes_nothing(store: MemoryStore):
    memory_id = store.add(category="fact", title="Old title", body="Old body")
    ok = store.edit(memory_id)
    assert ok is False
    row = store.get(memory_id)
    assert row["title"] == "Old title"


def test_edit_missing_id_returns_false(store: MemoryStore):
    assert store.edit(9999, title="X") is False


# ---------------------------------------------------------------------------
# mark_retrieved
# ---------------------------------------------------------------------------


def test_mark_retrieved_bumps_count_and_sets_last_used(store: MemoryStore):
    memory_id = store.add(category="fact", title="A", body="a")
    assert store.get(memory_id)["retrieval_count"] == 0
    assert store.get(memory_id)["last_used_at"] is None

    store.mark_retrieved([memory_id])
    row = store.get(memory_id)
    assert row["retrieval_count"] == 1
    assert row["last_used_at"] is not None


def test_mark_retrieved_accumulates_across_calls(store: MemoryStore):
    memory_id = store.add(category="fact", title="A", body="a")
    store.mark_retrieved([memory_id])
    store.mark_retrieved([memory_id])
    assert store.get(memory_id)["retrieval_count"] == 2


def test_mark_retrieved_empty_list_is_a_noop(store: MemoryStore):
    memory_id = store.add(category="fact", title="A", body="a")
    store.mark_retrieved([])
    assert store.get(memory_id)["retrieval_count"] == 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_counts_by_category_and_source(store: MemoryStore):
    store.add(category="decision", title="A", body="a", source="explicit")
    store.add(category="decision", title="B", body="b", source="auto_extracted")
    store.add(category="mistake", title="C", body="c", source="explicit", pinned=True)

    stats = store.stats()
    assert stats["total"] == 3
    assert stats["active"] == 3
    assert stats["superseded"] == 0
    assert stats["pinned"] == 1
    assert stats["by_category"] == {"decision": 2, "mistake": 1}
    assert stats["by_source"] == {"explicit": 2, "auto_extracted": 1}


def test_stats_counts_superseded_separately_from_active(store: MemoryStore):
    old_id = store.add(category="decision", title="Old", body="x")
    new_id = store.add(category="decision", title="New", body="y")
    store.supersede(old_id, new_id)

    stats = store.stats()
    assert stats["total"] == 2
    assert stats["active"] == 1
    assert stats["superseded"] == 1
    # Superseded rows are excluded from by_category counts (same rule as all_active()).
    assert stats["by_category"] == {"decision": 1}


def test_stats_oldest_and_newest(store: MemoryStore):
    old_id = store.add(category="fact", title="First", body="x")
    new_id = store.add(category="fact", title="Second", body="y")

    stats = store.stats()
    assert stats["oldest"]["id"] == old_id
    assert stats["newest"]["id"] == new_id


def test_stats_empty_store(store: MemoryStore):
    stats = store.stats()
    assert stats["total"] == 0
    assert stats["oldest"] is None
    assert stats["newest"] is None
    assert stats["by_category"] == {}
