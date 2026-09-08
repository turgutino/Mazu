"""Tests for the BM25 + semantic blending in retrieval.py. The core thing worth
proving here: when semantic search is opted into, a memory phrased in completely
different words from the query -- something pure BM25 cannot find at all, since it
requires shared vocabulary -- can still surface, purely from embedding similarity.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import mazu.memory.retrieval as retrieval
from mazu.memory.retrieval import (
    _rank_by_relevance,
    build_context_block,
    build_global_context_block,
    explain_retrieval,
)
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


# ---------------------------------------------------------------------------
# retrieval tracking (retrieval_count / last_used_at)
# ---------------------------------------------------------------------------


def test_build_context_block_marks_rendered_memories_as_retrieved(store: MemoryStore, monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    memory_id = store.add(category="decision", title="Use PostgreSQL", body="for the database")

    assert store.get(memory_id)["retrieval_count"] == 0
    build_context_block(store, query="what database do we use")
    row = store.get(memory_id)
    assert row["retrieval_count"] == 1
    assert row["last_used_at"] is not None


def test_build_context_block_does_not_mark_unrendered_memories(store: MemoryStore, monkeypatch):
    """A memory that exists but has zero relevance to the query and isn't a pinned/
    mistake floor entry shouldn't have its retrieval stats touched -- it was never
    actually shown to the model.
    """
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    shown_id = store.add(category="decision", title="Use PostgreSQL", body="for the database")
    unrelated_id = store.add(category="decision", title="Adopted React", body="for frontend components")

    build_context_block(store, query="what database do we use")

    assert store.get(shown_id)["retrieval_count"] == 1
    assert store.get(unrelated_id)["retrieval_count"] == 0


def test_build_context_block_empty_store_is_a_noop_for_tracking(store: MemoryStore):
    # Must not raise even though mark_retrieved([]) is called with nothing to update.
    block = build_context_block(store, query="anything")
    assert block == ""


def test_build_global_context_block_marks_retrieved(tmp_path: Path):
    global_store = MemoryStore(tmp_path / "global.db")
    memory_id = global_store.add(category="user_preference", title="Name", body="Turgut")

    build_global_context_block(global_store)

    assert global_store.get(memory_id)["retrieval_count"] == 1
    global_store.close()


# ---------------------------------------------------------------------------
# explain_retrieval (mazu memory why)
# ---------------------------------------------------------------------------


def test_explain_retrieval_never_mutates_retrieval_stats(store: MemoryStore, monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    memory_id = store.add(category="decision", title="Use PostgreSQL", body="for the database")

    explain_retrieval(store, query="what database do we use")

    assert store.get(memory_id)["retrieval_count"] == 0


def test_explain_retrieval_marks_pinned_as_always_included(store: MemoryStore, monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    store.add(category="fact", title="Pinned fact", body="x", pinned=True)

    explanations = explain_retrieval(store, query="something completely unrelated")

    pinned_entry = next(e for e in explanations if e["row"]["title"] == "Pinned fact")
    assert pinned_entry["included"] is True
    assert pinned_entry["reason"] == "pinned"


def test_explain_retrieval_marks_recent_mistake_as_always_included(store: MemoryStore, monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    store.add(category="mistake", title="Forgot to handle None", body="x")

    explanations = explain_retrieval(store, query="something completely unrelated")

    entry = next(e for e in explanations if e["row"]["title"] == "Forgot to handle None")
    assert entry["included"] is True
    assert entry["reason"] == "recent mistake"


def test_explain_retrieval_marks_low_relevance_as_not_included(store: MemoryStore, monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    store.add(category="decision", title="Use PostgreSQL", body="for the database")
    store.add(category="decision", title="Use React", body="for the frontend components")

    explanations = explain_retrieval(store, query="what database do we use", limit=1)

    included = [e for e in explanations if e["included"]]
    not_included = [e for e in explanations if not e["included"]]
    assert any(e["row"]["title"] == "Use PostgreSQL" for e in included)
    assert any(e["row"]["title"] == "Use React" for e in not_included)


def test_explain_retrieval_reports_bm25_score(store: MemoryStore, monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    store.add(category="decision", title="Use PostgreSQL", body="for the database")

    explanations = explain_retrieval(store, query="what database do we use")

    entry = next(e for e in explanations if e["row"]["title"] == "Use PostgreSQL")
    assert entry["bm25"] is not None
    assert entry["bm25"] > 0
    assert entry["combined"] is not None


def test_explain_retrieval_empty_store_returns_empty_list(store: MemoryStore):
    assert explain_retrieval(store, query="anything") == []


# ---------------------------------------------------------------------------
# CONTEXT_CHAR_BUDGET vs. floor rows (pinned / recent mistakes) -- real bug
# found live: ALL rows, floor included, shared one running char_count across
# every category in whatever order categories happened to be encountered, so
# an earlier-processed category's ranked entries could exhaust the budget
# before a LATER category's pinned row was ever reached -- silently dropping
# it, while `mazu memory why` kept claiming (unconditionally) that every
# pinned row WOULD be included regardless.
# ---------------------------------------------------------------------------


def test_pinned_memory_in_a_later_processed_category_is_never_dropped(store: MemoryStore, monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    monkeypatch.setattr(retrieval, "CONTEXT_CHAR_BUDGET", 400)

    # Pinned OLDER (so store.pinned()'s created_at DESC order processes "fact"
    # first) -- an unrelated pin just to occupy a second, later-processed
    # category.
    convention_id = store.add(category="convention", title="convention pin", body="c" * 20, pinned=True)
    fact_id = store.add(category="fact", title="fact pin", body="f" * 20, pinned=True)
    # Bulk ranked (non-pinned) "fact" entries -- rendered right after the fact
    # pin, in the SAME category block (floor rows always precede ranked rows
    # of the same category) -- deliberately large enough to blow well past the
    # 400-char budget on their own.
    for i in range(5):
        store.add(category="fact", title=f"filler {i}", body="y" * 200)

    block = build_context_block(store, query="")
    assert f"[id {fact_id}]" in block
    assert f"[id {convention_id}]" in block  # the row that used to get silently dropped

    explanations = explain_retrieval(store, query="")
    convention_entry = next(e for e in explanations if e["row"]["id"] == convention_id)
    assert convention_entry["included"] is True
    assert convention_entry["reason"] == "pinned"


def test_a_ranked_row_cut_for_budget_is_reported_as_such_by_explain_retrieval(
    store: MemoryStore, monkeypatch
):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    monkeypatch.setattr(retrieval, "CONTEXT_CHAR_BUDGET", 300)

    # all_active() orders most-recently-added first, and with no query
    # _rank_by_relevance falls back to that same recency order -- so the
    # LATER-added row here (kept_id) is ranked/rendered BEFORE cut_id.
    cut_id = store.add(category="fact", title="a much longer fact", body="z" * 400)
    kept_id = store.add(category="fact", title="short fact", body="short body")

    block = build_context_block(store, query="")
    assert f"[id {kept_id}]" in block
    assert f"[id {cut_id}]" not in block

    explanations = explain_retrieval(store, query="")
    cut_entry = next(e for e in explanations if e["row"]["id"] == cut_id)
    # Before this fix, explain_retrieval never modeled the char budget at all,
    # so a row within the top-`limit` candidates was always reported
    # "included" even when build_context_block would actually cut it.
    assert cut_entry["included"] is False
    assert cut_entry["reason"] == "cut for char budget"
