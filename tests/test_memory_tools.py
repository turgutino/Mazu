from pathlib import Path
from unittest.mock import patch

import pytest

from mazu.memory.store import MemoryStore
from mazu.tools.memory_tools import make_memory_tools


@pytest.fixture
def stores(tmp_path: Path):
    project = MemoryStore(tmp_path / "project.db")
    glob = MemoryStore(tmp_path / "global.db")
    yield project, glob
    project.close()
    glob.close()


def _tools(stores):
    project, glob = stores
    return {t.name: t for t in make_memory_tools(project, glob, session_id="s1")}


def test_remember_stores_in_project_store_by_default(stores):
    tools = _tools(stores)
    result = tools["remember"].handler(
        {"category": "decision", "title": "Use PostgreSQL", "body": "for concurrency"}
    )
    assert not result.is_error

    project, glob = stores
    assert len(project.all_active()) == 1
    assert len(glob.all_active()) == 0


def test_remember_user_preference_routes_to_global_store(stores):
    tools = _tools(stores)
    result = tools["remember"].handler(
        {"category": "user_preference", "title": "Name", "body": "Turgut"}
    )
    assert not result.is_error

    project, glob = stores
    assert len(project.all_active()) == 0
    assert len(glob.all_active()) == 1


def test_remember_does_not_compute_embedding_when_semantic_search_disabled(stores, monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    tools = _tools(stores)
    tools["remember"].handler(
        {"category": "fact", "title": "Test", "body": "body text"}
    )

    project, _ = stores
    row = project.all_active()[0]
    assert row["embedding"] is None


def test_remember_computes_embedding_when_semantic_search_enabled(stores, monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    with patch("mazu.tools.memory_tools.embed_text", return_value=[0.1, 0.2, 0.3]):
        tools = _tools(stores)
        tools["remember"].handler({"category": "fact", "title": "Test", "body": "body text"})

    from mazu.memory.embeddings import deserialize_embedding

    project, _ = stores
    row = project.all_active()[0]
    assert deserialize_embedding(row["embedding"]) == [0.1, 0.2, 0.3]


def test_remember_with_supersedes_id_marks_old_memory(stores):
    tools = _tools(stores)
    project, _ = stores
    old_id = project.add(category="decision", title="Use MySQL", body="initial pick")

    tools["remember"].handler(
        {
            "category": "decision",
            "title": "Use PostgreSQL",
            "body": "switched for concurrency",
            "supersedes_id": old_id,
        }
    )

    assert old_id not in {r["id"] for r in project.all_active()}


def test_recall_searches_project_store(stores):
    tools = _tools(stores)
    project, _ = stores
    project.add(category="decision", title="Use PostgreSQL", body="for concurrency")

    result = tools["recall"].handler({"query": "PostgreSQL"})
    assert "Use PostgreSQL" in result.content


def test_recall_no_match_returns_friendly_message(stores):
    tools = _tools(stores)
    result = tools["recall"].handler({"query": "nonexistent topic"})
    assert "No matching memories" in result.content
