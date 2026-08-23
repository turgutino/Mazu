from pathlib import Path
from unittest.mock import patch

import pytest

from mazu.memory.belief import (
    MAX_CLASSIFICATIONS_PER_BATCH,
    MIN_SIMILARITY,
    decide_action,
    find_gray_zone_match,
    run_shadow_belief_batch,
)
from mazu.memory.belief_log import BeliefShadowLogStore
from mazu.memory.store import FUZZY_DUPLICATE_THRESHOLD, MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


@pytest.fixture
def belief_log(tmp_path: Path) -> BeliefShadowLogStore:
    log = BeliefShadowLogStore(tmp_path / "belief_shadow.db")
    yield log
    log.close()


# ---------------------------------------------------------------------------
# Phase 1: schema migration
# ---------------------------------------------------------------------------


def test_needs_review_column_exists_and_defaults_false(store: MemoryStore):
    mid = store.add(category="fact", title="A", body="B")
    row = store.get(mid)
    assert row["needs_review"] == 0


def test_needs_review_migration_on_pre_existing_database(tmp_path):
    import sqlite3

    db_path = tmp_path / "old_memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE memories ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
        "category TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL, tags TEXT, "
        "source TEXT NOT NULL, session_id TEXT, relevance_score REAL NOT NULL DEFAULT 1.0, "
        "superseded_by INTEGER, pinned INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, task_summary TEXT);"
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path)
    try:
        mid = store.add(category="fact", title="A", body="B")
        assert store.get(mid)["needs_review"] == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Phase 2: find_gray_zone_match -- pure, local, zero API calls
# ---------------------------------------------------------------------------


def test_no_existing_memories_returns_none(store: MemoryStore):
    match, score = find_gray_zone_match(store, "fact", "Project uses SQLite", "for local dev")
    assert match is None
    assert score == 0.0


def test_clearly_unrelated_memory_returns_none(store: MemoryStore):
    store.add(category="fact", title="Deploys run on Fridays", body="team convention, no relation to db")
    match, score = find_gray_zone_match(store, "fact", "Project uses SQLite", "for local dev")
    assert match is None
    assert score < MIN_SIMILARITY


def test_exact_duplicate_is_not_gray_zone(store: MemoryStore):
    store.add(category="fact", title="Project uses PostgreSQL", body="for production concurrency")
    match, score = find_gray_zone_match(
        store, "fact", "Project uses PostgreSQL", "for production concurrency"
    )
    # Identical text scores 1.0 similarity -- above FUZZY_DUPLICATE_THRESHOLD, so
    # this is MemoryStore.find_duplicate's job, not a gray-zone judgment call.
    assert match is None
    assert score >= FUZZY_DUPLICATE_THRESHOLD


def test_fuzzy_duplicate_rephrasing_is_not_gray_zone(store: MemoryStore):
    store.add(category="convention", title="Functions use snake_case", body="applies to all Python functions")
    match, score = find_gray_zone_match(
        store, "convention", "Use snake_case for Python functions", "applies to all functions"
    )
    assert match is None
    assert score >= FUZZY_DUPLICATE_THRESHOLD


def test_gray_zone_match_found_between_thresholds(store: MemoryStore):
    old_id = store.add(category="decision", title="Authentication uses JWT", body="stateless tokens, no server sessions")
    match, score = find_gray_zone_match(
        store, "decision", "Authentication uses OAuth2", "delegated auth via provider"
    )
    assert match is not None
    assert match["id"] == old_id
    assert MIN_SIMILARITY <= score < FUZZY_DUPLICATE_THRESHOLD


def test_different_category_never_matches(store: MemoryStore):
    store.add(category="decision", title="Authentication uses JWT", body="stateless tokens")
    match, score = find_gray_zone_match(
        store, "mistake", "Authentication uses OAuth2", "delegated auth via provider"
    )
    assert match is None


def test_superseded_memory_excluded_from_gray_zone_search(store: MemoryStore):
    old_id = store.add(category="decision", title="Authentication uses JWT", body="stateless tokens, no sessions")
    new_id = store.add(category="decision", title="Authentication uses OAuth2", body="delegated auth")
    store.supersede(old_id, new_id)
    match, score = find_gray_zone_match(
        store, "decision", "Authentication uses OAuth2 with PKCE", "delegated auth with PKCE flow"
    )
    # Only the still-active OAuth2 row (new_id) should be considered, never the superseded JWT row.
    assert match is None or match["id"] != old_id


def test_archived_memory_excluded_from_gray_zone_search(store: MemoryStore):
    old_id = store.add(category="fact", title="Redis is used for caching", body="shared cache layer")
    store.archive(old_id)
    match, score = find_gray_zone_match(
        store, "fact", "Redis is temporarily disabled during development", "disabled locally for now"
    )
    assert match is None


def test_best_match_chosen_among_multiple_candidates(store: MemoryStore):
    unrelated_id = store.add(category="fact", title="CI runs on GitHub Actions", body="unrelated to caching")
    best_id = store.add(category="fact", title="Redis is used for caching", body="shared cache layer for sessions")
    match, score = find_gray_zone_match(
        store, "fact", "Redis is temporarily disabled during development", "disabled locally for now, caching off"
    )
    assert match is not None
    assert match["id"] == best_id


# ---------------------------------------------------------------------------
# decide_action -- pure decision table, no I/O
# ---------------------------------------------------------------------------


def test_unrelated_inserts_new_no_supersede():
    assert decide_action("unrelated", "high") == ("insert_new", False)


def test_duplicate_skips_no_supersede():
    assert decide_action("duplicate", "high") == ("skip_duplicate", False)


def test_refinement_high_confidence_supersedes():
    assert decide_action("refinement", "high") == ("supersede_old_with_new", True)


@pytest.mark.parametrize("confidence", ["low", "medium"])
def test_refinement_low_medium_confidence_does_not_supersede(confidence):
    action, would_supersede = decide_action("refinement", confidence)
    assert would_supersede is False
    assert action == "insert_new_keep_both"


def test_replacement_high_confidence_supersedes():
    assert decide_action("replacement", "high") == ("supersede_old_with_new", True)


@pytest.mark.parametrize("confidence", ["low", "medium"])
def test_replacement_low_medium_confidence_flags_for_review_not_supersede(confidence):
    action, would_supersede = decide_action("replacement", confidence)
    assert would_supersede is False
    assert action == "flag_for_review"


@pytest.mark.parametrize("confidence", ["low", "medium", "high"])
def test_contradiction_never_supersedes_regardless_of_confidence(confidence):
    # The single most important safety property of the whole decision table:
    # a "contradiction" classification is NEVER allowed to trigger a supersede,
    # no matter how confident the classifier claims to be.
    action, would_supersede = decide_action("contradiction", confidence)
    assert would_supersede is False
    assert action == "flag_for_review"


def test_temporary_state_never_supersedes():
    action, would_supersede = decide_action("temporary_state", "high")
    assert would_supersede is False
    assert action == "insert_new_keep_both_tag_temporary"


def test_unknown_relationship_never_supersedes():
    action, would_supersede = decide_action("something_new_and_unexpected", "high")
    assert would_supersede is False
    assert action == "flag_for_review"


# ---------------------------------------------------------------------------
# Phase 3: run_shadow_belief_batch -- classification + logging, SHADOW MODE ONLY
# ---------------------------------------------------------------------------


def _classify(relationship="contradiction", confidence="medium", reasoning="because"):
    return ({"relationship": relationship, "confidence": confidence, "reasoning": reasoning}, {})


def test_no_gray_zone_match_never_calls_classifier(store: MemoryStore, belief_log: BeliefShadowLogStore):
    items = [{"category": "fact", "title": "Totally unrelated new fact", "body": "nothing like anything else"}]
    with patch("mazu.memory.belief.classify_relationship") as mock_classify:
        results = run_shadow_belief_batch(store, belief_log, "s1", items)
    mock_classify.assert_not_called()
    assert results == []
    assert belief_log.all_decisions() == []


def test_gray_zone_match_triggers_exactly_one_classification_call(store: MemoryStore, belief_log: BeliefShadowLogStore):
    store.add(category="decision", title="Authentication uses JWT", body="stateless tokens, no server sessions")
    items = [{"category": "decision", "title": "Authentication uses OAuth2", "body": "delegated auth via provider"}]
    with patch("mazu.memory.belief.classify_relationship", return_value=_classify()) as mock_classify:
        results = run_shadow_belief_batch(store, belief_log, "s1", items)
    mock_classify.assert_called_once()
    assert len(results) == 1


def test_shadow_batch_never_mutates_the_memory_store(store: MemoryStore, belief_log: BeliefShadowLogStore):
    old_id = store.add(category="fact", title="Project uses PostgreSQL", body="for concurrency and durability")
    items = [{"category": "fact", "title": "Project now uses SQLite", "body": "for local dev simplicity"}]
    with patch(
        "mazu.memory.belief.classify_relationship",
        return_value=_classify(relationship="replacement", confidence="high"),
    ):
        run_shadow_belief_batch(store, belief_log, "s1", items)

    # Even though replacement+high WOULD supersede in a live system, the actual
    # store must be completely untouched: no new row inserted, old row not superseded.
    row = store.get(old_id)
    assert row["superseded_by"] is None
    assert row["title"] == "Project uses PostgreSQL"
    assert row["body"] == "for concurrency and durability"
    assert len(store.all_active()) == 1


def test_shadow_batch_never_sets_needs_review_automatically(store: MemoryStore, belief_log: BeliefShadowLogStore):
    old_id = store.add(category="fact", title="Redis is used for caching", body="shared cache layer for sessions")
    items = [{"category": "fact", "title": "Redis is temporarily disabled during development", "body": "disabled locally, caching off for now"}]
    with patch(
        "mazu.memory.belief.classify_relationship",
        return_value=_classify(relationship="contradiction", confidence="high"),
    ):
        run_shadow_belief_batch(store, belief_log, "s1", items)
    assert store.get(old_id)["needs_review"] == 0


@pytest.mark.parametrize(
    "relationship,confidence,expected_action,expected_would_supersede",
    [
        ("unrelated", "high", "insert_new", False),
        ("duplicate", "high", "skip_duplicate", False),
        ("refinement", "high", "supersede_old_with_new", True),
        ("refinement", "low", "insert_new_keep_both", False),
        ("replacement", "high", "supersede_old_with_new", True),
        ("replacement", "medium", "flag_for_review", False),
        ("contradiction", "high", "flag_for_review", False),
        ("temporary_state", "medium", "insert_new_keep_both_tag_temporary", False),
    ],
)
def test_shadow_batch_logs_expected_decision_for_each_scenario(
    store, belief_log, relationship, confidence, expected_action, expected_would_supersede
):
    store.add(category="fact", title="Authentication uses JWT", body="stateless tokens, no server sessions")
    items = [{"category": "fact", "title": "Authentication uses OAuth2", "body": "delegated auth via provider"}]
    with patch(
        "mazu.memory.belief.classify_relationship",
        return_value=_classify(relationship=relationship, confidence=confidence),
    ):
        run_shadow_belief_batch(store, belief_log, "s1", items)

    logged = belief_log.all_decisions()
    assert len(logged) == 1
    row = logged[0]
    assert row["relationship"] == relationship
    assert row["confidence"] == confidence
    assert row["proposed_action"] == expected_action
    assert bool(row["would_supersede"]) == expected_would_supersede
    assert row["classifier_failed"] == 0


def test_classifier_failure_is_logged_and_does_not_raise(store: MemoryStore, belief_log: BeliefShadowLogStore):
    store.add(category="fact", title="Authentication uses JWT", body="stateless tokens, no server sessions")
    items = [{"category": "fact", "title": "Authentication uses OAuth2", "body": "delegated auth via provider"}]
    with patch("mazu.memory.belief.classify_relationship", side_effect=RuntimeError("API down")):
        results = run_shadow_belief_batch(store, belief_log, "s1", items)

    logged = belief_log.all_decisions()
    assert len(logged) == 1
    assert logged[0]["classifier_failed"] == 1
    assert logged[0]["proposed_action"] == "classifier_failed"
    assert "API down" in logged[0]["reasoning"]
    # And the memory store is still untouched.
    assert len(store.all_active()) == 1


def test_classifier_returning_invalid_relationship_is_treated_as_failure(store, belief_log):
    store.add(category="fact", title="Authentication uses JWT", body="stateless tokens, no server sessions")
    items = [{"category": "fact", "title": "Authentication uses OAuth2", "body": "delegated auth via provider"}]
    with patch(
        "mazu.memory.belief.classify_relationship",
        return_value=({"relationship": "not_a_real_relationship", "confidence": "high"}, {}),
    ):
        run_shadow_belief_batch(store, belief_log, "s1", items)
    logged = belief_log.all_decisions()
    assert logged[0]["classifier_failed"] == 1
    assert logged[0]["proposed_action"] == "classifier_failed"


def test_max_classifications_per_batch_is_enforced(store: MemoryStore, belief_log: BeliefShadowLogStore):
    # 3 existing memories, each with a corresponding gray-zone-related new item --
    # cap at 2 to prove the bound is enforced independent of extraction's own 0-5 limit.
    for i in range(3):
        store.add(category="fact", title=f"Authentication uses JWT variant {i}", body="stateless tokens, no sessions")
    items = [
        {"category": "fact", "title": f"Authentication uses OAuth2 variant {i}", "body": "delegated auth via provider"}
        for i in range(3)
    ]
    with patch("mazu.memory.belief.classify_relationship", return_value=_classify()) as mock_classify:
        results = run_shadow_belief_batch(store, belief_log, "s1", items, max_classifications=2)

    assert mock_classify.call_count == 2
    logged = belief_log.all_decisions()
    assert len(logged) == 3
    actions = [row["proposed_action"] for row in logged]
    assert actions.count("skipped_cap") == 1


def test_default_max_classifications_matches_module_constant():
    assert MAX_CLASSIFICATIONS_PER_BATCH == 5


def test_empty_extracted_items_does_nothing(store: MemoryStore, belief_log: BeliefShadowLogStore):
    with patch("mazu.memory.belief.classify_relationship") as mock_classify:
        results = run_shadow_belief_batch(store, belief_log, "s1", [])
    mock_classify.assert_not_called()
    assert results == []
    assert belief_log.all_decisions() == []


# ---------------------------------------------------------------------------
# Integration: mazu.agent.session.insert_extracted_memories wiring
# ---------------------------------------------------------------------------


def test_insert_extracted_memories_without_belief_log_is_unchanged(store: MemoryStore):
    """The most important backward-compatibility guarantee: every existing
    caller that doesn't pass belief_log gets byte-for-byte today's behavior."""
    from mazu.agent.session import insert_extracted_memories

    store.add(category="decision", title="Authentication uses JWT", body="stateless tokens, no server sessions")
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "decision", "title": "Authentication uses OAuth2", "body": "delegated auth"}], {}),
    ), patch("mazu.memory.belief.classify_relationship") as mock_classify:
        inserted, extracted = insert_extracted_memories(store, "s1", [{"role": "user", "content": "hi"}])

    mock_classify.assert_not_called()
    assert inserted == 1  # existing dedup/insert behavior fully intact


def test_insert_extracted_memories_with_belief_log_still_inserts_normally(
    store: MemoryStore, belief_log: BeliefShadowLogStore
):
    """Passing belief_log adds shadow-mode observation but must not change what
    actually gets inserted -- the real insert path is completely independent."""
    from mazu.agent.session import insert_extracted_memories

    store.add(category="decision", title="Authentication uses JWT", body="stateless tokens, no server sessions")
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "decision", "title": "Authentication uses OAuth2", "body": "delegated auth"}], {}),
    ), patch(
        "mazu.memory.belief.classify_relationship",
        return_value=_classify(relationship="replacement", confidence="high"),
    ):
        inserted, extracted = insert_extracted_memories(
            store, "s1", [{"role": "user", "content": "hi"}], belief_log=belief_log
        )

    assert inserted == 1  # same insert outcome as without belief_log
    logged = belief_log.all_decisions()
    assert len(logged) == 1
    assert logged[0]["proposed_action"] == "supersede_old_with_new"  # logged...
    # ...but never actually applied:
    old_row = store.search(category="decision")
    assert all(r["superseded_by"] is None for r in old_row)


def test_insert_extracted_memories_belief_shadow_failure_does_not_block_insert(
    store: MemoryStore, belief_log: BeliefShadowLogStore
):
    from mazu.agent.session import insert_extracted_memories

    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "fact", "title": "Brand new fact", "body": "no relation to anything"}], {}),
    ), patch("mazu.agent.session.run_shadow_belief_batch", side_effect=RuntimeError("shadow blew up")):
        inserted, extracted = insert_extracted_memories(
            store, "s1", [{"role": "user", "content": "hi"}], belief_log=belief_log
        )

    assert inserted == 1  # the real insert must succeed regardless of shadow-mode failures
