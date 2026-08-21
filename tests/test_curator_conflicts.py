import pytest

from mazu.curator.analysis.conflicts import find_candidate_pairs
from mazu.curator.store import CuratorStore, new_curator_run_id
from mazu.curator.tools.conflicts import make_curator_conflict_tools
from mazu.curator.tools.registry import CURATOR_FORBIDDEN_TOOLS, build_curator_registry
from mazu.memory.store import MemoryStore


@pytest.fixture
def memory_store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


@pytest.fixture
def curator_store(tmp_path):
    s = CuratorStore(tmp_path / "curator.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# find_candidate_pairs
# ---------------------------------------------------------------------------


def test_no_pairs_when_store_is_empty(memory_store):
    assert find_candidate_pairs(memory_store) == []


def test_near_identical_memories_are_excluded_as_duplicates_not_conflicts(memory_store):
    """Exact duplicates are consolidate.py's job -- the conflicts band deliberately
    excludes similarity above MAX_SIMILARITY."""
    memory_store.add(category="fact", title="the sky is blue", body="the sky is blue today")
    memory_store.add(category="fact", title="the sky is blue", body="the sky is blue today")
    assert find_candidate_pairs(memory_store) == []


def test_unrelated_memories_are_excluded(memory_store):
    memory_store.add(category="fact", title="database is postgres", body="we use postgres for storage")
    memory_store.add(category="fact", title="ci runs on github actions", body="tests run via github actions")
    assert find_candidate_pairs(memory_store) == []


def test_related_but_distinct_memories_are_candidates(memory_store):
    memory_store.add(
        category="decision", title="use postgres for storage",
        body="chose postgres because of jsonb support and strong community tooling",
    )
    memory_store.add(
        category="decision", title="use mysql for storage",
        body="chose mysql for the team familiarity and existing ops tooling",
    )
    pairs = find_candidate_pairs(memory_store)
    assert len(pairs) == 1


def test_different_categories_are_never_paired(memory_store):
    memory_store.add(category="decision", title="use postgres for storage", body="chose postgres over mysql")
    memory_store.add(category="mistake", title="use postgres for storage", body="chose postgres over mysql")
    # Same text, different category -- pairing only happens within a category.
    pairs = find_candidate_pairs(memory_store)
    assert pairs == []


def test_archived_and_superseded_memories_are_excluded(memory_store):
    old_id = memory_store.add(
        category="decision", title="use postgres for storage",
        body="chose postgres because of jsonb support and strong community tooling",
    )
    new_id = memory_store.add(
        category="decision", title="use mysql for storage",
        body="chose mysql for the team familiarity and existing ops tooling",
    )
    memory_store.supersede(old_id, new_id)
    assert find_candidate_pairs(memory_store) == []


def test_max_pairs_caps_the_result(memory_store):
    for i in range(6):
        memory_store.add(category="decision", title=f"use tool {i} for X", body=f"chose tool {i} over the others for X")
    pairs = find_candidate_pairs(memory_store, max_pairs=2)
    assert len(pairs) <= 2


# ---------------------------------------------------------------------------
# CuratorStore conflict methods
# ---------------------------------------------------------------------------


def test_record_and_list_conflicts(curator_store):
    conflict_id = curator_store.record_conflict(1, 2, "contradiction", "these disagree", confidence=0.8)
    rows = curator_store.list_conflicts()
    assert len(rows) == 1
    assert rows[0]["id"] == conflict_id
    assert rows[0]["resolved_at"] is None


def test_resolve_conflict_marks_resolved_and_excludes_from_unresolved_list(curator_store):
    conflict_id = curator_store.record_conflict(1, 2, "contradiction", "these disagree")
    ok = curator_store.resolve_conflict(conflict_id, "superseded 1 with 2")
    assert ok is True
    assert curator_store.list_conflicts(unresolved_only=True) == []
    all_rows = curator_store.list_conflicts(unresolved_only=False)
    assert len(all_rows) == 1
    assert all_rows[0]["resolution"] == "superseded 1 with 2"


def test_resolve_missing_conflict_returns_false(curator_store):
    assert curator_store.resolve_conflict(9999, "x") is False


# ---------------------------------------------------------------------------
# Conflict tools
# ---------------------------------------------------------------------------


def test_record_conflict_tool_rejects_invalid_kind(memory_store, curator_store):
    run_id = new_curator_run_id()
    tools = {t.name: t for t in make_curator_conflict_tools(memory_store, curator_store, run_id, dry_run=False)}
    result = tools["record_conflict"].handler({
        "memory_id_a": 1, "memory_id_b": 2, "kind": "bogus", "rationale": "x",
    })
    assert result.is_error


def test_record_conflict_tool_rejects_nonexistent_memory_ids(memory_store, curator_store):
    """Regression test: find_related_memory_pairs only ever surfaces real,
    existing memory ids, so a well-behaved model can never trigger this -- but
    nothing structurally validated it, so a stray/hallucinated id could have
    left memory_conflicts with a dangling reference."""
    run_id = new_curator_run_id()
    real_id = memory_store.add(category="fact", title="real", body="body")
    tools = {t.name: t for t in make_curator_conflict_tools(memory_store, curator_store, run_id, dry_run=False)}

    result = tools["record_conflict"].handler({
        "memory_id_a": real_id, "memory_id_b": 999999, "kind": "unrelated", "rationale": "x",
    })
    assert result.is_error
    assert curator_store.list_conflicts(unresolved_only=False) == []


def test_conflicts_area_registry_has_no_forbidden_tools(memory_store, curator_store, tmp_path):
    run_id = new_curator_run_id()
    registry = build_curator_registry(
        area="conflicts", run_id=run_id, dry_run=False,
        memory_store=memory_store, curator_store=curator_store, root=tmp_path,
    )
    registered_names = {schema["name"] for schema in registry.schemas()}
    assert registered_names & CURATOR_FORBIDDEN_TOOLS == set()
    assert "read_file" in registered_names
    assert "list_dir" in registered_names
    assert "glob_files" in registered_names


def test_conflicts_area_requires_memory_store_and_root(curator_store):
    run_id = new_curator_run_id()
    with pytest.raises(ValueError, match="requires memory_store"):
        build_curator_registry(area="conflicts", run_id=run_id, dry_run=False, curator_store=curator_store)


def test_memory_tools_used_from_within_a_conflicts_pass_are_logged_as_conflicts(memory_store, curator_store, tmp_path):
    """Regression test for a real bug found live: memory tools (archive_memory,
    supersede_memory, etc.) are reused inside the conflicts area's registry so a
    genuine contradiction can actually be resolved, not just recorded. Those
    tools used to hardcode area='memory' internally regardless of which
    orchestrator area invoked them -- so a real resolution made during a
    `mazu curator run --area conflicts` pass was silently mis-tagged and
    invisible to `mazu curator log --area conflicts`. Confirmed live: a real
    supersede_memory call made while resolving a genuine contradiction showed up
    under `--area memory` instead of `--area conflicts`.
    """
    run_id = new_curator_run_id()
    old_id = memory_store.add(category="decision", title="use aws", body="chose aws")
    new_id = memory_store.add(category="decision", title="use gcp", body="moved to gcp")

    registry = build_curator_registry(
        area="conflicts", run_id=run_id, dry_run=False,
        memory_store=memory_store, curator_store=curator_store, root=tmp_path,
    )
    registry.get("supersede_memory").handler({
        "old_id": old_id, "new_id": new_id, "rationale": "resolves the aws/gcp contradiction",
    })

    conflicts_entries = curator_store.log_recent(area="conflicts")
    memory_entries = curator_store.log_recent(area="memory")
    assert len(conflicts_entries) == 1
    assert conflicts_entries[0]["action"] == "supersede_memory"
    assert memory_entries == []


def test_config_tools_used_from_housekeeping_are_logged_as_housekeeping(curator_store, tmp_path):
    """Same bug class, same fix, for the housekeeping area's config tools (which
    used to hardcode area='config')."""
    from mazu.config import set_config_value

    run_id = new_curator_run_id()
    registry = build_curator_registry(
        area="housekeeping", run_id=run_id, dry_run=False, curator_store=curator_store, root=tmp_path,
    )
    registry.get("set_config").handler({
        "key": "router_suggestions", "value": "false", "rationale": "reduce noise",
    })

    housekeeping_entries = curator_store.log_recent(area="housekeeping")
    config_entries = curator_store.log_recent(area="config")
    assert len(housekeeping_entries) == 1
    assert housekeeping_entries[0]["action"] == "set_config"
    assert config_entries == []
