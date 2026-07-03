from mazu.agent.context import build_system_prompt
from mazu.agent.interaction import safe_confirm
from mazu.agent.session import finalize_session
from mazu.banner import print_banner
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.client import _split_model, default_model, run_turn, summarize_usage
from mazu.llm.errors import MazuAPIError
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager
from mazu.tools.registry import ToolRegistry
from mazu.tools.shell import is_denied_shell_command


def _confirm(tool_name: str, tool_input: dict) -> bool:
    print(f"\n[confirm] {tool_name}({tool_input})")
    return safe_confirm("Run this? [y/N] ")


def run_chat_loop(
    registry: ToolRegistry,
    session_id: str,
    memory_store: MemoryStore | None = None,
    global_memory_store: MemoryStore | None = None,
    skill_manager: SkillManager | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    model: str | None = None,
) -> None:
    messages: list[dict] = []
    system_prompt = None  # built lazily from the first real task, so retrieval has a query

    print_banner()
    provider_name, model_name = _split_model(model or default_model())
    print(f"model: {provider_name}:{model_name}")
    print(
        "chat — type your task. Commands: /checkpoint, /rollback [id]. Ctrl+C to quit.\n"
    )
    try:
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return
            if not user_input:
                continue

            if user_input == "/checkpoint":
                _handle_checkpoint(checkpoint_manager, messages)
                continue

            if user_input == "/rollback" or user_input.startswith("/rollback "):
                new_messages = _handle_rollback(checkpoint_manager, user_input, messages)
                if new_messages is not None:
                    messages[:] = new_messages
                continue

            if system_prompt is None:
                if memory_store is not None:
                    memory_store.start_session(session_id)
                system_prompt = build_system_prompt(
                    memory_store, skill_manager, query=user_input, global_memory_store=global_memory_store
                )
                if system_prompt.strip() != "":
                    print("[memory] loaded prior context relevant to this task\n")

            messages.append({"role": "user", "content": user_input})
            _run_until_done(messages, registry, system_prompt, model)
    finally:
        if memory_store is not None:
            finalize_session(memory_store, session_id, messages, model=model)
        if global_memory_store is not None:
            global_memory_store.close()


def _handle_checkpoint(checkpoint_manager: CheckpointManager | None, messages: list[dict]) -> None:
    if checkpoint_manager is None:
        print("Checkpointing is not available.")
        return
    entry = checkpoint_manager.snapshot(messages, trigger="manual")
    print(f"[checkpoint] {entry['id']} saved (commit {entry['git_commit'][:8]})")


def _handle_rollback(
    checkpoint_manager: CheckpointManager | None, user_input: str, messages: list[dict]
) -> list[dict] | None:
    if checkpoint_manager is None:
        print("Rollback is not available.")
        return None
    parts = user_input.split(maxsplit=1)
    checkpoint_id = parts[1].strip() if len(parts) > 1 else None
    try:
        entry, diff_stat = checkpoint_manager.preview_rollback(checkpoint_id)
    except ValueError as e:
        print(str(e))
        return None

    print(f"\nRolling back to {entry['id']} ({entry['created_at']}): {entry['summary']}")
    if diff_stat.strip():
        print(diff_stat)
    if not safe_confirm(
        "This discards code changes, memory writes, and conversation state since then. Proceed? [y/N] "
    ):
        print("Rollback cancelled.")
        return None

    result = checkpoint_manager.restore(entry["id"])
    print(f"[rollback] restored to {entry['id']}. Code, memory, and conversation state reset.")
    return result["messages"]


def _run_until_done(
    messages: list[dict], registry: ToolRegistry, system_prompt: str, model: str | None
) -> None:
    while True:
        try:
            response = run_turn(messages, system_prompt, registry.schemas(), model=model)
        except MazuAPIError as e:
            print(f"\n[error] {e}\nReturning to the prompt — try again, or use /rollback.")
            return
        messages.append({"role": "assistant", "content": response.content})
        print(f"[usage] {summarize_usage(response.usage)}")

        text_blocks = [b["text"] for b in response.content if b["type"] == "text"]
        if text_blocks:
            print("\n".join(text_blocks))

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block["type"] != "tool_use":
                continue
            tool = registry.get(block["name"])
            if tool is None:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": f"Unknown tool: {block['name']}",
                        "is_error": True,
                    }
                )
                continue
            if tool.name == "run_shell" and is_denied_shell_command(
                block["input"].get("command", "")
            ):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "Blocked: command matches the safety denylist.",
                        "is_error": True,
                    }
                )
                continue
            if tool.destructive and not _confirm(tool.name, block["input"]):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "User declined to run this tool.",
                        "is_error": True,
                    }
                )
                continue
            result = tool.handler(block["input"])
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})
