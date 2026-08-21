import pytest

from mazu.curator.store import CuratorStore, new_curator_run_id
from mazu.curator.tools.registry import CURATOR_FORBIDDEN_TOOLS, build_curator_registry
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager


@pytest.fixture
def curator_store(tmp_path):
    s = CuratorStore(tmp_path / "curator.db")
    yield s
    s.close()


@pytest.fixture
def memory_store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


@pytest.fixture
def skill_manager(tmp_path):
    return SkillManager(tmp_path)


def test_memory_area_registry_contains_no_forbidden_tools(curator_store, memory_store):
    """The single most important test in the whole feature: the built registry's
    tool names must be disjoint from CURATOR_FORBIDDEN_TOOLS, under every area.
    This -- not the system prompt -- is what actually guarantees Curator can never
    write outside .mazu/, run shell, run arbitrary saved skills, or reuse
    remember()'s embedding-call side effect on the user's own OpenAI key."""
    run_id = new_curator_run_id()
    registry = build_curator_registry(
        area="memory", run_id=run_id, dry_run=False,
        memory_store=memory_store, curator_store=curator_store,
    )
    registered_names = {schema["name"] for schema in registry.schemas()}
    assert registered_names & CURATOR_FORBIDDEN_TOOLS == set()


def test_memory_area_registry_contains_expected_tools(curator_store, memory_store):
    run_id = new_curator_run_id()
    registry = build_curator_registry(
        area="memory", run_id=run_id, dry_run=False,
        memory_store=memory_store, curator_store=curator_store,
    )
    registered_names = {schema["name"] for schema in registry.schemas()}
    expected = {
        "curator_note", "recall_memories", "list_memories", "memory_stats",
        "find_duplicate_memories", "find_stale_memories", "add_memory",
        "edit_memory", "archive_memory", "unarchive_memory", "pin_memory",
        "unpin_memory", "supersede_memory",
    }
    assert registered_names == expected


def test_forbidden_tools_constant_covers_the_real_risk_names():
    """A cheap guard against someone quietly shrinking the forbidden set later --
    these five names must always be present."""
    assert CURATOR_FORBIDDEN_TOOLS == frozenset({
        "write_file", "edit_file", "run_shell", "run_skill", "remember",
    })


def test_unknown_area_raises(curator_store, memory_store):
    run_id = new_curator_run_id()
    with pytest.raises(ValueError, match="Unknown or not-yet-implemented"):
        build_curator_registry(
            area="bogus", run_id=run_id, dry_run=False,
            memory_store=memory_store, curator_store=curator_store,
        )


def test_memory_area_requires_memory_store(curator_store):
    run_id = new_curator_run_id()
    with pytest.raises(ValueError, match="requires memory_store"):
        build_curator_registry(area="memory", run_id=run_id, dry_run=False, curator_store=curator_store)


def test_skills_area_registry_contains_no_forbidden_tools(curator_store, skill_manager):
    run_id = new_curator_run_id()
    registry = build_curator_registry(
        area="skills", run_id=run_id, dry_run=False,
        skill_manager=skill_manager, curator_store=curator_store,
    )
    registered_names = {schema["name"] for schema in registry.schemas()}
    assert registered_names & CURATOR_FORBIDDEN_TOOLS == set()
    assert "run_skill" not in registered_names


def test_skills_area_registry_contains_expected_tools(curator_store, skill_manager):
    run_id = new_curator_run_id()
    registry = build_curator_registry(
        area="skills", run_id=run_id, dry_run=False,
        skill_manager=skill_manager, curator_store=curator_store,
    )
    registered_names = {schema["name"] for schema in registry.schemas()}
    expected = {
        "curator_note", "list_skills", "read_skill", "skill_outcomes",
        "write_skill", "archive_skill", "unarchive_skill", "supersede_skill",
    }
    assert registered_names == expected


def test_skills_area_requires_skill_manager(curator_store):
    run_id = new_curator_run_id()
    with pytest.raises(ValueError, match="requires skill_manager"):
        build_curator_registry(area="skills", run_id=run_id, dry_run=False, curator_store=curator_store)


def test_housekeeping_area_registry_contains_no_forbidden_tools(curator_store, tmp_path):
    run_id = new_curator_run_id()
    registry = build_curator_registry(
        area="housekeeping", run_id=run_id, dry_run=False, curator_store=curator_store, root=tmp_path,
    )
    registered_names = {schema["name"] for schema in registry.schemas()}
    assert registered_names & CURATOR_FORBIDDEN_TOOLS == set()


def test_housekeeping_area_registry_contains_expected_tools(curator_store, tmp_path):
    run_id = new_curator_run_id()
    registry = build_curator_registry(
        area="housekeeping", run_id=run_id, dry_run=False, curator_store=curator_store, root=tmp_path,
    )
    registered_names = {schema["name"] for schema in registry.schemas()}
    expected = {
        "curator_note", "run_stats", "usage_summary", "find_abandoned_branches",
        "action_log_patterns", "get_config", "set_config",
    }
    assert registered_names == expected


def test_housekeeping_area_requires_root(curator_store):
    run_id = new_curator_run_id()
    with pytest.raises(ValueError, match="requires root"):
        build_curator_registry(area="housekeeping", run_id=run_id, dry_run=False, curator_store=curator_store)
