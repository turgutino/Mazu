from mazu.agent.prompts import SYSTEM_PROMPT
from mazu.memory.retrieval import build_context_block, build_global_context_block
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager


def build_system_prompt(
    memory_store: MemoryStore | None,
    skill_manager: SkillManager | None,
    query: str,
    global_memory_store: MemoryStore | None = None,
) -> str:
    parts = [SYSTEM_PROMPT]
    if global_memory_store is not None:
        block = build_global_context_block(global_memory_store)
        if block:
            parts.append(block)
    if memory_store is not None:
        block = build_context_block(memory_store, query=query)
        if block:
            parts.append(block)
    if skill_manager is not None:
        block = skill_manager.build_context_block()
        if block:
            parts.append(block)
    return "\n\n".join(parts)
