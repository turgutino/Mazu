"""Regression tests for a real bug found live: `mazu run --dry-run` claims
"nothing was actually changed", but its `remember`/`save_skill`/`run_skill`
tools (mazu/tools/memory_tools.py, mazu/tools/skill_tools.py) and the
end-of-session memory extraction (mazu/agent/session.py's finalize_session)
had no dry_run concept at all -- unlike write_file/edit_file/run_shell, which
already did. A dry-run session could permanently write to memory.db, write a
real skill file, execute arbitrary previously-saved Python, and make a real
billed LLM call, all while reporting itself as a no-op preview.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from mazu.agent.registry_factory import build_registry
from mazu.agent.session import finalize_session
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager
from mazu.tools.memory_tools import make_memory_tools
from mazu.tools.skill_tools import make_skill_tools


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


@pytest.fixture
def global_store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "global_memory.db")
    yield s
    s.close()


def test_remember_under_dry_run_does_not_write_to_memory_store(store, global_store):
    tools = {
        t.name: t
        for t in make_memory_tools(store, global_store, "session-1", dry_run=True)
    }
    result = tools["remember"].handler(
        {"category": "decision", "title": "Use PostgreSQL", "body": "For concurrency"}
    )
    assert not result.is_error
    assert "[dry-run]" in result.content
    assert "Would remember" in result.content
    assert store.search() == []


def test_remember_under_dry_run_reports_the_would_be_supersede(store, global_store):
    existing_id = store.add(category="decision", title="Use SQLite", body="Simpler")
    tools = {
        t.name: t
        for t in make_memory_tools(store, global_store, "session-1", dry_run=True)
    }
    result = tools["remember"].handler(
        {
            "category": "decision", "title": "Use PostgreSQL", "body": "For concurrency",
            "supersedes_id": existing_id,
        }
    )
    assert f"would mark memory {existing_id} as superseded" in result.content
    # Nothing actually superseded -- the original memory is still active.
    assert store.get(existing_id)["superseded_by"] is None


def test_remember_without_dry_run_still_writes_for_real(store, global_store):
    # Default is False -- proves this is additive, not a behavior change for
    # every existing caller that never passes dry_run at all.
    tools = {t.name: t for t in make_memory_tools(store, global_store, "session-1")}
    result = tools["remember"].handler(
        {"category": "decision", "title": "Use PostgreSQL", "body": "For concurrency"}
    )
    assert "[dry-run]" not in result.content
    assert len(store.search()) == 1


def test_save_skill_under_dry_run_does_not_write_a_skill_file(tmp_path):
    manager = SkillManager(tmp_path)
    tools = {t.name: t for t in make_skill_tools(manager, dry_run=True)}
    result = tools["save_skill"].handler(
        {"name": "double", "description": "doubles a number", "code": "def run(args):\n    return str(args['n'] * 2)"}
    )
    assert not result.is_error
    assert "[dry-run]" in result.content
    assert manager.list() == []


def test_run_skill_under_dry_run_does_not_execute_anything(tmp_path):
    manager = SkillManager(tmp_path)
    manager.save(name="double", description="doubles a number", code="def run(args):\n    return str(args['n'] * 2)")
    tools = {t.name: t for t in make_skill_tools(manager, dry_run=True)}
    result = tools["run_skill"].handler({"name": "double", "args": {"n": 5}})
    assert not result.is_error
    assert "[dry-run]" in result.content
    assert "10" not in result.content  # the real skill was never actually run


def test_finalize_session_under_dry_run_skips_extraction_entirely(store):
    with patch("mazu.agent.session.extract_memories") as mock_extract:
        inserted = finalize_session(
            store, "session-1", messages=[{"role": "user", "content": "hi"}], dry_run=True,
        )
    mock_extract.assert_not_called()
    assert inserted == 0

    # finalize_session() closes `store` internally (same real behavior in every
    # mode) -- reopen to confirm nothing was written.
    reopened = MemoryStore(store.db_path)
    assert reopened.search() == []
    reopened.close()


def test_build_registry_threads_dry_run_into_memory_and_skill_tools(tmp_path, store, global_store):
    # End-to-end proof, at the same seam mazu/cli.py's `run` command actually
    # uses: build_registry(dry_run=True) must reach ALL four dry-run-aware tool
    # factories (fs, shell -- already correct before this fix -- and now memory,
    # skills too), not just the two that happened to be wired originally.
    manager = SkillManager(tmp_path)
    registry = build_registry(tmp_path, store, global_store, manager, "session-1", dry_run=True)

    result = registry.get("remember").handler({"category": "fact", "title": "x", "body": "y"})
    assert "[dry-run]" in result.content
    assert store.search() == []

    result = registry.get("save_skill").handler(
        {"name": "noop", "description": "does nothing", "code": "def run(args):\n    return ''"}
    )
    assert "[dry-run]" in result.content
    assert manager.list() == []
