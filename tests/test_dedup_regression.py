from pathlib import Path
from unittest.mock import patch

import pytest

from mazu.agent.session import finalize_session
from mazu.memory.extraction import EXTRACTION_INSTRUCTIONS
from mazu.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


def test_extraction_prompt_excludes_personal_facts():
    """Regression test for a real bug found via live testing: auto-extraction used to
    save personal facts (the user's name/age) into project-scoped memory instead of the
    global user_preference store. The fix was a prompt change telling the extraction
    model not to do that -- this pins the instruction text so it can't silently regress.
    """
    assert "personal facts about the person" in EXTRACTION_INSTRUCTIONS
    assert "Do NOT extract" in EXTRACTION_INSTRUCTIONS


def test_finalize_session_skips_exact_duplicate(store: MemoryStore):
    # Simulate the explicit `remember` tool already having saved this fact...
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency")

    # ...then auto-extraction re-derives the same fact from the transcript.
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=[
            {"category": "decision", "title": "Use PostgreSQL", "body": "For concurrency"}
        ],
    ):
        finalize_session(store, "session-1", messages=[{"role": "user", "content": "hi"}])

    store2 = MemoryStore(store.db_path)
    rows = store2.search(category="decision")
    store2.close()
    assert len(rows) == 1  # not duplicated


def test_finalize_session_inserts_genuinely_new_memory(store: MemoryStore):
    with patch(
        "mazu.agent.session.extract_memories",
        return_value=[{"category": "convention", "title": "Use snake_case", "body": "for functions"}],
    ):
        finalize_session(store, "session-1", messages=[{"role": "user", "content": "hi"}])

    store2 = MemoryStore(store.db_path)
    rows = store2.search()
    store2.close()
    assert len(rows) == 1
    assert rows[0]["source"] == "auto_extracted"


def test_finalize_session_handles_extraction_failure_gracefully(store: MemoryStore, capsys):
    with patch("mazu.agent.session.extract_memories", side_effect=RuntimeError("API down")):
        finalize_session(store, "session-1", messages=[{"role": "user", "content": "hi"}])

    store2 = MemoryStore(store.db_path)
    rows = store2.search()
    store2.close()
    assert rows == []  # no crash, no partial data
    assert "extraction skipped" in capsys.readouterr().out
