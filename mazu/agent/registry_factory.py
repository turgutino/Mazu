from pathlib import Path

from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager
from mazu.tools.fs import make_fs_tools
from mazu.tools.memory_tools import make_memory_tools
from mazu.tools.registry import ToolRegistry
from mazu.tools.shell import make_shell_tool
from mazu.tools.skill_tools import make_skill_tools


def build_registry(
    root: Path,
    memory_store: MemoryStore,
    global_memory_store: MemoryStore,
    skill_manager: SkillManager,
    session_id: str,
    dry_run: bool = False,
    belief_log=None,
) -> ToolRegistry:
    """Shared by mazu/cli.py (chat/run/council) and mazu/agent/explore.py -- pulled
    out of cli.py so explore.py (which needs to build one registry per worktree-
    rooted branch) can reuse it without cli.py importing explore.py and explore.py
    importing cli.py back (a circular import).

    `belief_log`, when given, is forwarded to make_memory_tools() so the
    `remember` tool's real, explicit actions (insert/supersede) get SHADOW-ONLY
    observed (mazu.memory.belief_log) -- purely additive logging, never changes
    what `remember` actually does. Defaults to None: omitting it reproduces
    today's exact behavior for every existing caller.
    """
    registry = ToolRegistry()
    for tool in make_fs_tools(root, dry_run=dry_run):
        registry.register(tool)
    registry.register(make_shell_tool(root, dry_run=dry_run))
    for tool in make_memory_tools(
        memory_store, global_memory_store, session_id, belief_log=belief_log, project=str(root)
    ):
        registry.register(tool)
    for tool in make_skill_tools(skill_manager):
        registry.register(tool)
    return registry
