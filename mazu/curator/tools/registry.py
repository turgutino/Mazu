from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.curator.store import CuratorStore
from mazu.curator.tools.analysis import make_curator_analysis_tools
from mazu.curator.tools.config_tools import make_curator_config_tools
from mazu.curator.tools.conflicts import make_curator_conflict_tools
from mazu.curator.tools.diary import make_diary_tools
from mazu.curator.tools.memory import make_curator_memory_tools
from mazu.curator.tools.skills import make_curator_skill_tools
from mazu.memory.store import MemoryStore
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.tools.fs import make_fs_tools
from mazu.tools.registry import ToolRegistry
from mazu.usage.store import UsageStore

# Read-only subset of make_fs_tools -- the only filesystem access any Curator
# area ever gets (the 'conflicts' area, to check a memory's claim against actual
# code), same READ_ONLY philosophy as council._read_only_registry.
_READ_ONLY_FS_TOOL_NAMES = {"read_file", "list_dir", "glob_files"}

# Tools that must NEVER appear in a Curator registry, under any area, under any
# circumstance -- this is the actual enforcement mechanism for "Curator never
# touches user project source / never runs arbitrary shell / never re-implements
# remember() with its embedding-call side effect on the user's own OpenAI key" --
# not the system prompt. build_curator_registry() asserts this at build time;
# tests/test_curator_registry.py asserts it again independently.
CURATOR_FORBIDDEN_TOOLS = frozenset({
    "write_file", "edit_file", "run_shell", "run_skill", "remember",
})


def build_curator_registry(
    area: str,
    run_id: str,
    dry_run: bool,
    memory_store: MemoryStore | None = None,
    curator_store: CuratorStore | None = None,
    skill_manager: SkillManager | None = None,
    action_log_store: ActionLogStore | None = None,
    root: Path | None = None,
    run_store: RunStore | None = None,
    usage_store: UsageStore | None = None,
) -> ToolRegistry:
    """Builds a FRESH ToolRegistry containing only the tools relevant to `area`,
    never mazu.agent.registry_factory.build_registry (which unconditionally
    includes filesystem/shell tools). Supports area in
    {'memory', 'skills', 'housekeeping', 'conflicts'}.
    """
    registry = ToolRegistry()

    if curator_store is not None:
        for tool in make_diary_tools(curator_store, run_id, area):
            registry.register(tool)

    if area == "memory":
        if memory_store is None or curator_store is None:
            raise ValueError("area='memory' requires memory_store and curator_store")
        for tool in make_curator_memory_tools(memory_store, curator_store, run_id, dry_run, area=area):
            registry.register(tool)
    elif area == "skills":
        if skill_manager is None or curator_store is None:
            raise ValueError("area='skills' requires skill_manager and curator_store")
        for tool in make_curator_skill_tools(skill_manager, curator_store, run_id, dry_run, action_log_store):
            registry.register(tool)
    elif area == "housekeeping":
        if curator_store is None or root is None:
            raise ValueError("area='housekeeping' requires root and curator_store")
        for tool in make_curator_analysis_tools(root, run_store, usage_store, action_log_store):
            registry.register(tool)
        for tool in make_curator_config_tools(curator_store, run_id, dry_run, area=area):
            registry.register(tool)
    elif area == "conflicts":
        if memory_store is None or curator_store is None or root is None:
            raise ValueError("area='conflicts' requires memory_store, root, and curator_store")
        # Reuses the memory area's own tools so a genuine contradiction can
        # actually be resolved (edit/supersede/archive), not just recorded --
        # area=area here is the fix for a real bug found live: without it, every
        # such resolution made during a conflicts pass got silently logged under
        # area='memory' instead (see make_curator_memory_tools' docstring).
        for tool in make_curator_memory_tools(memory_store, curator_store, run_id, dry_run, area=area):
            registry.register(tool)
        for tool in make_curator_conflict_tools(memory_store, curator_store, run_id, dry_run):
            registry.register(tool)
        for tool in make_fs_tools(root, dry_run=False):
            if tool.name in _READ_ONLY_FS_TOOL_NAMES:
                registry.register(tool)
    else:
        raise ValueError(f"Unknown or not-yet-implemented curator area: '{area}'")

    registered_names = {schema["name"] for schema in registry.schemas()}
    forbidden_present = registered_names & CURATOR_FORBIDDEN_TOOLS
    assert not forbidden_present, (
        f"Curator registry for area '{area}' contains forbidden tool(s): {forbidden_present}"
    )
    return registry
