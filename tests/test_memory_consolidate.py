from pathlib import Path

import pytest

from mazu.memory.consolidate import apply_consolidation, find_duplicate_clusters
from mazu.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


def test_no_duplicates_finds_nothing(store: MemoryStore):
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency")
    store.add(category="decision", title="Use React", body="For component reuse")

    clusters = find_duplicate_clusters(store)
    assert clusters == []


def test_finds_simple_duplicate_pair(store: MemoryStore):
    store.add(
        category="decision",
        title="Use PostgreSQL for the database",
        body="Chosen for concurrency and JSON support",
    )
    store.add(
        category="decision",
        title="Database is PostgreSQL",
        body="Chosen for concurrency and JSON support",
    )

    clusters = find_duplicate_clusters(store)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_does_not_cluster_unrelated_memories(store: MemoryStore):
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency and JSON")
    store.add(category="mistake", title="Forgot to add an index", body="Query got slow at scale")

    clusters = find_duplicate_clusters(store)
    assert clusters == []


def test_does_not_cluster_across_categories_even_with_similar_wording(store: MemoryStore):
    # Same vocabulary, but a decision and a fact about the same topic are not
    # duplicates of each other -- they're different kinds of statements.
    store.add(
        category="decision", title="Use PostgreSQL for storage", body="concurrency JSON support"
    )
    store.add(category="fact", title="PostgreSQL storage in use", body="concurrency JSON support")

    clusters = find_duplicate_clusters(store)
    assert clusters == []


def test_transitive_chain_clusters_all_three(store: MemoryStore):
    # A and C share almost no words directly, but both overlap heavily with B --
    # union-find should still merge all three into one cluster.
    store.add(
        category="fact",
        title="alpha bravo charlie delta echo",
        body="foxtrot golf hotel india juliet",
    )
    store.add(
        category="fact",
        title="alpha bravo charlie delta echo",
        body="kilo lima mike november oscar",
    )
    store.add(
        category="fact",
        title="alpha bravo charlie delta echo",
        body="papa quebec romeo sierra tango",
    )

    clusters = find_duplicate_clusters(store, threshold=0.3)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_already_superseded_memory_excluded(store: MemoryStore):
    old_id = store.add(category="decision", title="Use MySQL", body="Initial pick")
    new_id = store.add(category="decision", title="Use PostgreSQL", body="Better fit")
    store.supersede(old_id, new_id)

    dup_id = store.add(category="decision", title="Use PostgreSQL now", body="Better fit")

    clusters = find_duplicate_clusters(store)
    # Only the active PostgreSQL pair should cluster -- the superseded MySQL entry
    # must not be considered at all.
    assert len(clusters) == 1
    ids_in_cluster = {r["id"] for r in clusters[0]}
    assert ids_in_cluster == {new_id, dup_id}
    assert old_id not in ids_in_cluster


def test_dry_run_does_not_modify_store(store: MemoryStore):
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency and JSON")
    store.add(category="decision", title="PostgreSQL for storage", body="For concurrency and JSON")

    find_duplicate_clusters(store)  # read-only call

    assert len(store.all_active()) == 2  # nothing was superseded


def test_apply_consolidation_keeps_newest_as_survivor(store: MemoryStore):
    old_id = store.add(category="decision", title="Use PostgreSQL", body="For concurrency and JSON")
    new_id = store.add(
        category="decision", title="PostgreSQL for storage", body="For concurrency and JSON"
    )

    clusters = find_duplicate_clusters(store)
    summary = apply_consolidation(store, clusters)

    assert len(summary) == 1
    assert summary[0]["survivor_id"] == new_id
    assert summary[0]["superseded"] == [{"id": old_id, "title": "Use PostgreSQL"}]

    active_ids = {r["id"] for r in store.all_active()}
    assert active_ids == {new_id}


def test_apply_consolidation_transitive_cluster_leaves_exactly_one_survivor(store: MemoryStore):
    id_a = store.add(category="fact", title="alpha bravo charlie", body="foxtrot golf hotel")
    id_b = store.add(category="fact", title="alpha bravo charlie", body="kilo lima mike")
    id_c = store.add(category="fact", title="alpha bravo charlie", body="papa quebec romeo")

    clusters = find_duplicate_clusters(store, threshold=0.15)
    apply_consolidation(store, clusters)

    active = store.all_active()
    assert len(active) == 1
    assert active[0]["id"] == id_c  # newest of the three
    assert id_a not in {r["id"] for r in active}
    assert id_b not in {r["id"] for r in active}


def test_higher_threshold_finds_fewer_clusters(store: MemoryStore):
    store.add(category="fact", title="alpha bravo charlie", body="delta echo foxtrot")
    store.add(category="fact", title="alpha bravo golf", body="hotel india juliet")

    loose = find_duplicate_clusters(store, threshold=0.1)
    strict = find_duplicate_clusters(store, threshold=0.9)

    assert len(loose) >= len(strict)
