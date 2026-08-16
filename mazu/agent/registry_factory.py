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
) -> ToolRegistry:
    """Shared by mazu/cli.py (chat/run/council) and mazu/agent/explore.py -- pulled
    out of cli.py so explore.py (which needs to build one registry per worktree-
    rooted branch) can reuse it without cli.py importing explore.py and explore.py
    importing cli.py back (a circular import).
    """
    registry = ToolRegistry()
    for tool in make_fs_tools(root, dry_run=dry_run):
        registry.register(tool)
    registry.register(make_shell_tool(root, dry_run=dry_run))
    for tool in make_memory_tools(memory_store, global_memory_store, session_id):
        registry.register(tool)
    for tool in make_skill_tools(skill_manager):
        registry.register(tool)
    return registry
