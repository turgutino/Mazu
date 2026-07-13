"""Tests for the BM25 + semantic blending in retrieval.py. The core thing worth
proving here: when semantic search is opted into, a memory phrased in completely
different words from the query -- something pure BM25 cannot find at all, since it
requires shared vocabulary -- can still surface, purely from embedding similarity.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from mazu.memory.retrieval import _rank_by_relevance, build_context_block
from mazu.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


def test_bm25_only_when_semantic_not_available(monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    pool = [
        {"id": 1, "title": "We use PostgreSQL", "body": "as our database", "tags": ""},
        {"id": 2, "title": "React chosen", "body": "for frontend components", "tags": ""},
    ]
    result = _rank_by_relevance(pool, "what database do we use", limit=10)
    # No shared vocabulary with memory 2 at all -- pure BM25 (semantic disabled)
    # must exclude it entirely.
    assert [r["id"] for r in result] == [1]


def test_semantic_match_recovers_memory_bm25_alone_would_miss(monkeypatch):
    """The core case semantic search exists for: zero shared vocabulary between the
    query and the memory, so BM25's score is 0 and it would normally be excluded
    entirely -- but a stored embedding with high cosine similarity should still
    surface it.
    """
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    # Deliberately no word overlap at all with the query below.
    pool = [
        {
            "id": 1,
            "title": "Storage engine choice",
            "body": "Postgres selected for ACID guarantees",
            "tags": "",
            "embedding": '[1.0, 0.0, 0.0]',
        },
        {
            "id": 2,
            "title": "Unrelated frontend note",
            "body": "component library picked",
            "tags": "",
            "embedding": '[0.0, 1.0, 0.0]',
        },
    ]

    def _fake_embed_text(text):
        # The query embeds identically to memory 1's stored vector -- maximum
        # cosine similarity -- and orthogonally (zero similarity) to memory 2's.
        return [1.0, 0.0, 0.0]

    with patch("mazu.memory.retrieval.embed_text", side_effect=_fake_embed_text):
        result = _rank_by_relevance(pool, "completely different wording entirely", limit=10)

    assert result, "semantic layer should have surfaced at least the matching memory"
    assert result[0]["id"] == 1


def test_semantic_blend_still_respects_bm25_when_both_signal(monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    pool = [
        {
            "id": 1,
            "title": "Use PostgreSQL",
            "body": "for the database, chosen for concurrency",
            "tags": "",
            "embedding": '[1.0, 0.0]',
        },
        {
            "id": 2,
            "title": "Use React",
            "body": "for the frontend, chosen for component reuse",
            "tags": "",
            "embedding": '[0.0, 1.0]',
        },
    ]

    with patch("mazu.memory.retrieval.embed_text", return_value=[1.0, 0.0]):
        result = _rank_by_relevance(pool, "what database does this project use", limit=10)

    # Both keyword overlap (BM25) and semantic similarity agree here -- the
    # PostgreSQL memory should rank first either way.
    assert result[0]["id"] == 1


def test_memories_without_stored_embeddings_still_rank_via_bm25(monkeypatch):
    """Not every memory will have an embedding (older rows from before semantic
    search was enabled, or written while it was off) -- those must not be silently
    dropped, just fall back to their BM25 contribution alone.
    """
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    pool = [
        {"id": 1, "title": "Use PostgreSQL", "body": "for the database", "tags": "", "embedding": None},
    ]

    with patch("mazu.memory.retrieval.embed_text", return_value=[1.0, 0.0]):
        result = _rank_by_relevance(pool, "what database do we use", limit=10)

    assert [r["id"] for r in result] == [1]


def test_build_context_block_works_end_to_end_with_semantic_disabled(store: MemoryStore, monkeypatch):
    """Regression safety: with semantic search off (the default), context building
    must behave exactly as it did before this feature existed.
    """
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    store.add(category="decision", title="Use PostgreSQL", body="for the database")

    block = build_context_block(store, query="what database do we use")
    assert "Use PostgreSQL" in block


def test_build_context_block_with_semantic_enabled_and_real_stored_embedding(
    store: MemoryStore, monkeypatch
):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    store.add(
        category="decision",
        title="Storage engine",
        body="Postgres chosen for ACID guarantees",
        embedding=[1.0, 0.0],
    )

    with patch("mazu.memory.retrieval.embed_text", return_value=[1.0, 0.0]):
        block = build_context_block(store, query="totally unrelated wording")

    assert "Storage engine" in block
