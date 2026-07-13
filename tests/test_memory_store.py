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
