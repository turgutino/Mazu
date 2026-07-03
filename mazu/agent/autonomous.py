from mazu.agent.context import build_system_prompt
from mazu.agent.interaction import safe_confirm
from mazu.agent.session import finalize_session
from mazu.banner import print_banner
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.client import _split_model, default_model, run_turn, summarize_usage
from mazu.llm.errors import MazuAPIError
from mazu.llm.pricing import estimate_cost
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager
from mazu.tools.registry import ToolRegistry
from mazu.tools.shell import is_denied_shell_command


def run_autonomous(
    registry: ToolRegistry,
    task: str,
    session_id: str,
    checkpoint_manager: CheckpointManager,
    memory_store: MemoryStore | None = None,
    global_memory_store: MemoryStore | None = None,
    skill_manager: SkillManager | None = None,
    max_steps: int = 15,
    checkpoint_every: int = 1,
    allow_shell: bool = False,
    max_consecutive_failures: int = 3,
    max_cost: float | None = None,
    model: str | None = None,
) -> None:
    if checkpoint_manager.is_dirty():
        print(
            "Working tree has uncommitted changes. `mazu run` requires a clean baseline "
            "so checkpoints are meaningful. Commit your changes first (or run `mazu checkpoint`)."
        )
        return

    if memory_store is not None:
        memory_store.start_session(session_id)
    system_prompt = build_system_prompt(
        memory_store, skill_manager, query=task, global_memory_store=global_memory_store
    )

    provider_name, model_name = _split_model(model or default_model())
    resolved_model = f"{provider_name}:{model_name}"
    cost_trackable = estimate_cost(resolved_model, 0, 0) is not None
    if max_cost is not None and not cost_trackable:
        print(
            f"[cost] warning: no pricing data for '{resolved_model}' — --max-cost can't be "
            "enforced for this model and will be ignored."
        )

    messages: list[dict] = [{"role": "user", "content": task}]
    print_banner()
    print(f"model: {resolved_model}")
    print(
        f"run — task: {task}\n"
        f"max-steps={max_steps} checkpoint-every={checkpoint_every} allow-shell={allow_shell}"
        f"{f' max-cost=${max_cost:.2f}' if max_cost is not None and cost_trackable else ''}\n"
    )

    step = 0
    consecutive_failures = 0
    total_in = 0
    total_out = 0
    total_cost = 0.0
    try:
        while step < max_steps:
            step += 1
            try:
                try:
                    response = run_turn(messages, system_prompt, registry.schemas(), model=model)
                except MazuAPIError as e:
                    print(f"\n[error] {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(
                            f"\nStopping: {max_consecutive_failures} consecutive failures "
                            "(including API errors)."
                        )
                        break
                    continue

                messages.append({"role": "assistant", "content": response.content})

                usage = response.usage
                step_in = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                step_out = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                total_in += step_in
                total_out += step_out
                cost_suffix = ""
                if cost_trackable:
                    total_cost += estimate_cost(resolved_model, step_in, step_out) or 0.0
                    cost_suffix = f" | ~${total_cost:.4f} so far"
                print(
                    f"[usage] step: {summarize_usage(usage)} | "
                    f"running total: {total_in} in, {total_out} out{cost_suffix}"
                )

                text_blocks = [b["text"] for b in response.content if b["type"] == "text"]
                if text_blocks:
                    print(f"[step {step}] " + "\n".join(text_blocks))

                if response.stop_reason != "tool_use":
                    print("\nTask complete (model signaled end_turn).")
                    break

                if max_cost is not None and cost_trackable and total_cost >= max_cost:
                    print(
                        f"\nStopping: estimated cost ~${total_cost:.4f} reached the "
                        f"--max-cost limit (${max_cost:.2f})."
                    )
                    break

                tool_results, round_failed = _execute_round(response, registry, allow_shell)
                messages.append({"role": "user", "content": tool_results})

                consecutive_failures = consecutive_failures + 1 if round_failed else 0
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\nStopping: {max_consecutive_failures} consecutive tool failures.")
                    break

                if step % checkpoint_every == 0:
                    summary = text_blocks[0][:100] if text_blocks else f"step {step}"
                    entry = checkpoint_manager.snapshot(
                        messages, trigger="auto_after_tool_round", summary=summary
                    )
                    print(f"[checkpoint {entry['id']} @ {entry['git_commit'][:8]}]")
            except KeyboardInterrupt:
                if not _handle_interrupt(checkpoint_manager, messages):
                    break
        else:
            print(f"\nStopped: reached max-steps ({max_steps}).")
    finally:
        if memory_store is not None:
            finalize_session(memory_store, session_id, messages, model=model)
        if global_memory_store is not None:
            global_memory_store.close()


def _execute_round(
    response, registry: ToolRegistry, allow_shell: bool
) -> tuple[list[dict], bool]:
    tool_results = []
    round_failed = False
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
            round_failed = True
            continue

        if tool.name == "run_shell":
            command = block["input"].get("command", "")
            if is_denied_shell_command(command):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "Blocked: command matches the safety denylist.",
                        "is_error": True,
                    }
                )
                round_failed = True
                continue
            if not allow_shell:
                print(f"\n[confirm] run_shell({block['input']})")
                if not safe_confirm("Run this? [y/N] "):
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
        if result.is_error:
            round_failed = True
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": result.content,
                "is_error": result.is_error,
            }
        )
    return tool_results, round_failed


def _handle_interrupt(checkpoint_manager: CheckpointManager, messages: list[dict]) -> bool:
    """Returns True to continue the run, False to stop it."""
    print("\n\nInterrupted.")
    while True:
        try:
            choice = input("[c]ontinue, [r]ollback <id>, [q]uit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n(no input available — quitting)")
            return False
        if choice == "c":
            return True
        if choice == "q":
            return False
        if choice.startswith("r"):
            parts = choice.split(maxsplit=1)
            checkpoint_id = parts[1] if len(parts) > 1 else None
            try:
                entry, _ = checkpoint_manager.preview_rollback(checkpoint_id)
            except ValueError as e:
                print(str(e))
                continue
            print(f"Rolling back to {entry['id']}: {entry['summary']}")
            result = checkpoint_manager.restore(entry["id"])
            messages[:] = result["messages"]
            print(f"Restored to {entry['id']}.")
            return True
