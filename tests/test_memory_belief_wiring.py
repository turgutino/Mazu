"""Tests that the real session-end and compaction call sites (mazu.agent.session,
mazu.agent.compaction) correctly activate shadow-mode belief logging when a
belief_log is supplied, and are byte-for-byte unchanged when it isn't. The
underlying classification/decision-table logic itself is tested exhaustively in
tests/test_memory_belief.py -- this file is specifically about the wiring:
does activating shadow mode through the real call paths actually work, and does
it ever put the real insert path at risk.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from mazu.agent.compaction import extract_memories_on_compaction
from mazu.agent.session import finalize_session, insert_extracted_memories
from mazu.memory.belief_log import BeliefShadowLogStore
from mazu.memory.store import MemoryStore


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


def _classify(relationship="replacement", confidence="high", reasoning="looks like a swap"):
    return ({"relationship": relationship, "confidence": confidence, "reasoning": reasoning}, {})


# ---------------------------------------------------------------------------
# 1. Real session-end extraction (finalize_session) activates shadow logging
# ---------------------------------------------------------------------------


def test_finalize_session_activates_shadow_logging_when_belief_log_given(store, belief_log):
    store.add(category="fact", title="Project uses PostgreSQL", body="for production concurrency")
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "fact", "title": "Project now uses SQLite", "body": "for local dev"}], {}),
    ), patch("mazu.memory.belief.classify_relationship", return_value=_classify()) as mock_classify:
        finalize_session(store, "s1", [{"role": "user", "content": "hi"}], belief_log=belief_log)

    mock_classify.assert_called_once()
    logged = belief_log.all_decisions()
    assert len(logged) == 1
    assert logged[0]["proposed_action"] == "supersede_old_with_new"


def test_finalize_session_without_belief_log_never_touches_classifier(store):
    """2 & 6: existing (non-shadow) behavior is fully preserved when belief_log
    is omitted -- the classifier must never even be imported/called."""
    store.add(category="fact", title="Project uses PostgreSQL", body="for production concurrency")
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "fact", "title": "Project now uses SQLite", "body": "for local dev"}], {}),
    ), patch("mazu.memory.belief.classify_relationship") as mock_classify:
        inserted = finalize_session(store, "s1", [{"role": "user", "content": "hi"}])

    mock_classify.assert_not_called()
    assert inserted == 1  # identical outcome to before this feature existed


def test_finalize_session_never_mutates_existing_memory_via_shadow_path(store, belief_log):
    """4: no real memory is mutated, even when the classifier confidently
    proposes a supersede."""
    old_id = store.add(category="fact", title="Project uses PostgreSQL", body="for production concurrency")
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "fact", "title": "Project now uses SQLite", "body": "for local dev"}], {}),
    ), patch(
        "mazu.memory.belief.classify_relationship",
        return_value=_classify(relationship="replacement", confidence="high"),
    ):
        finalize_session(store, "s1", [{"role": "user", "content": "hi"}], belief_log=belief_log)

    # finalize_session() closes `store` internally -- reopen to inspect final state.
    reopened = MemoryStore(store.db_path)
    old_row = reopened.get(old_id)
    reopened.close()
    assert old_row["superseded_by"] is None
    assert old_row["title"] == "Project uses PostgreSQL"
    assert old_row["needs_review"] == 0


def test_finalize_session_shadow_classifier_failure_never_blocks_insertion(store, belief_log):
    """3: shadow classifier failure never blocks memory insertion."""
    store.add(category="fact", title="Project uses PostgreSQL", body="for production concurrency")
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "fact", "title": "Project now uses SQLite", "body": "for local dev"}], {}),
    ), patch("mazu.memory.belief.classify_relationship", side_effect=RuntimeError("classifier down")):
        inserted = finalize_session(store, "s1", [{"role": "user", "content": "hi"}], belief_log=belief_log)

    assert inserted == 1  # the real fact still gets saved
    logged = belief_log.all_decisions()
    assert logged[0]["classifier_failed"] == 1


def test_finalize_session_shadow_infrastructure_failure_never_blocks_insertion(store, tmp_path):
    """A broken belief_log (or any exception inside the shadow batch call
    itself, not just the classifier) must not prevent the real insert either --
    session.py wraps the whole shadow call in its own try/except."""
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "fact", "title": "Brand new unrelated fact", "body": "nothing like anything"}], {}),
    ), patch("mazu.agent.session.run_shadow_belief_batch", side_effect=RuntimeError("belief_log unreachable")):
        inserted = finalize_session(
            store, "s1", [{"role": "user", "content": "hi"}],
            belief_log=BeliefShadowLogStore(tmp_path / "unused.db"),
        )
    assert inserted == 1


# ---------------------------------------------------------------------------
# Real compaction-time extraction (extract_memories_on_compaction) wiring
# ---------------------------------------------------------------------------


def test_compaction_callback_activates_shadow_logging_when_belief_log_given(store, belief_log):
    store.add(category="decision", title="Authentication uses JWT", body="stateless tokens, no server sessions")
    on_compacted = extract_memories_on_compaction(store, "s1", model=None, belief_log=belief_log)
    assert on_compacted is not None

    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "decision", "title": "Authentication uses OAuth2", "body": "delegated auth"}], {}),
    ), patch("mazu.memory.belief.classify_relationship", return_value=_classify()) as mock_classify:
        on_compacted([{"role": "user", "content": "some compacted-away chunk"}])

    mock_classify.assert_called_once()
    assert len(belief_log.all_decisions()) == 1


def test_compaction_callback_without_belief_log_never_touches_classifier(store):
    on_compacted = extract_memories_on_compaction(store, "s1", model=None)
    assert on_compacted is not None
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=([{"category": "fact", "title": "Some fact", "body": "some body"}], {}),
    ), patch("mazu.memory.belief.classify_relationship") as mock_classify:
        on_compacted([{"role": "user", "content": "chunk"}])
    mock_classify.assert_not_called()


def test_compaction_callback_none_memory_store_returns_none_regardless_of_belief_log(belief_log):
    assert extract_memories_on_compaction(None, "s1", model=None, belief_log=belief_log) is None


def test_compaction_callback_never_mutates_memory_via_shadow_path(store, belief_log):
    old_id = store.add(category="fact", title="Redis is used for caching", body="shared cache layer for sessions")
    on_compacted = extract_memories_on_compaction(store, "s1", model=None, belief_log=belief_log)

    with patch(
        "mazu.agent.session.extract_memories",
        return_value=(
            [{"category": "fact", "title": "Redis is temporarily disabled during development",
              "body": "disabled locally, caching off for now"}],
            {},
        ),
    ), patch(
        "mazu.memory.belief.classify_relationship",
        return_value=_classify(relationship="contradiction", confidence="high"),
    ):
        on_compacted([{"role": "user", "content": "chunk"}])

    old_row = store.get(old_id)
    assert old_row["superseded_by"] is None
    assert old_row["needs_review"] == 0


# ---------------------------------------------------------------------------
# 5. Maximum classification calls remain bounded through the real wired paths
# ---------------------------------------------------------------------------


def test_finalize_session_respects_classification_cap_through_real_wiring(store, belief_log):
    for i in range(6):
        store.add(category="fact", title=f"Authentication uses JWT variant {i}", body="stateless tokens, no sessions")
    extracted = [
        {"category": "fact", "title": f"Authentication uses OAuth2 variant {i}", "body": "delegated auth via provider"}
        for i in range(6)
    ]
    with patch("mazu.agent.session.extract_memories", return_value=(extracted, {})), patch(
        "mazu.memory.belief.classify_relationship", return_value=_classify()
    ) as mock_classify:
        finalize_session(store, "s1", [{"role": "user", "content": "hi"}], belief_log=belief_log)

    # extract_memories itself caps at 0-5 memories per its own schema, and the
    # shadow batch additionally hard-caps at MAX_CLASSIFICATIONS_PER_BATCH (5) --
    # 6 items here proves the batch-level cap holds even if a caller ever fed it more.
    from mazu.memory.belief import MAX_CLASSIFICATIONS_PER_BATCH

    assert mock_classify.call_count <= MAX_CLASSIFICATIONS_PER_BATCH
