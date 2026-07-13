from mazu.action_log.store import ActionLogStore, record_action
from mazu.agent.compaction import compact_if_needed, force_compact
from mazu.agent.context import build_system_prompt
from mazu.agent.interaction import safe_confirm
from mazu.agent.session import finalize_session
from mazu.banner import print_banner
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.client import _split_model, default_model, run_turn, summarize_usage
from mazu.llm.errors import MazuAPIError, MazuContextLengthError
from mazu.llm.pricing import estimate_cost
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager
from mazu.tools.registry import ToolRegistry
from mazu.tools.shell import denylist_reason, is_allowed_by_shell_allowlist
from mazu.usage.store import UsageStore


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
    usage_store: UsageStore | None = None,
    action_log_store: ActionLogStore | None = None,
    shell_allowlist: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    # The clean-baseline requirement exists so a real run's checkpoints are
    # meaningful diffs against a known-good starting point. A dry run never writes
    # a file, runs a command for real, or creates a checkpoint, so there's nothing
    # for a dirty tree to make ambiguous -- and refusing to preview a plan just
    # because you have other uncommitted work in progress would be actively unhelpful.
    if not dry_run and checkpoint_manager.is_dirty():
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
        f"{f' max-cost=${max_cost:.2f}' if max_cost is not None and cost_trackable else ''}"
        f"{' [DRY RUN — no files will be written, no commands will be run, no checkpoints will be created]' if dry_run else ''}\n"
    )

    step = 0
    consecutive_failures = 0
    total_in = 0
    total_out = 0
    total_cost = 0.0
    try:
        while step < max_steps:
            step += 1
            messages, compacted = compact_if_needed(messages, model)
            if compacted:
                print(f"[context] compacted conversation history ({len(messages)} messages remain)")
            try:
                try:
                    try:
                        response = run_turn(messages, system_prompt, registry.schemas(), model=model)
                    except MazuContextLengthError:
                        # The proactive char-based estimate can undershoot (very dense
                        # tool output, a single huge message) -- this is the safety net.
                        # Compact much more aggressively than the proactive pass and
                        # retry exactly once; if it still fails, the error falls through
                        # to the same MazuAPIError handling as any other failure below.
                        print(
                            "\n[context] hit the model's context limit — compacting "
                            "aggressively and retrying once"
                        )
                        messages = force_compact(messages, model)
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
                step_cost = estimate_cost(resolved_model, step_in, step_out) if cost_trackable else None
                cost_suffix = ""
                if step_cost is not None:
                    total_cost += step_cost
                    cost_suffix = f" | ~${total_cost:.4f} so far"
                if usage_store is not None:
                    usage_store.log(
                        "run", session_id, provider_name, model_name, step_in, step_out, step_cost
                    )
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

                tool_results, round_failed = _execute_round(
                    response, registry, allow_shell, action_log_store, session_id, shell_allowlist, dry_run
                )
                messages.append({"role": "user", "content": tool_results})

                consecutive_failures = consecutive_failures + 1 if round_failed else 0
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\nStopping: {max_consecutive_failures} consecutive tool failures.")
                    break

                if not dry_run and step % checkpoint_every == 0:
                    summary = text_blocks[0][:100] if text_blocks else f"step {step}"
                    entry = checkpoint_manager.snapshot(
                        messages, trigger="auto_after_tool_round", summary=summary
                    )
                    print(f"[checkpoint {entry['id']} @ {entry['git_commit'][:8]}]")
            except KeyboardInterrupt:
                if dry_run:
                    # Nothing was actually written, so there's nothing to preview a
                    # rollback of and no live conversation state to preserve across
                    # a restore -- just stop, matching the "no side effects" contract.
                    break
                if not _handle_interrupt(checkpoint_manager, messages):
                    break
        else:
            print(f"\nStopped: reached max-steps ({max_steps}).")
    finally:
        if memory_store is not None:
            finalize_session(memory_store, session_id, messages, model=model)
        if global_memory_store is not None:
            global_memory_store.close()
        if usage_store is not None:
            usage_store.close()
        if action_log_store is not None:
            action_log_store.close()


def _execute_round(
    response,
    registry: ToolRegistry,
    allow_shell: bool,
    action_log_store: ActionLogStore | None = None,
    session_id: str | None = None,
    shell_allowlist: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[dict], bool]:
    tool_results = []
    round_failed = False
    for block in response.content:
        if block["type"] != "tool_use":
            continue
        tool = registry.get(block["name"])
        if tool is None:
            record_action(
                action_log_store, session_id, "run", block["name"], block["input"],
                "unknown_tool", f"Unknown tool: {block['name']}",
            )
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
            reason = denylist_reason(command)
            if reason is not None:
                msg = f"Blocked: command {reason} (safety denylist)."
                record_action(action_log_store, session_id, "run", tool.name, block["input"], "blocked", msg)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block["id"], "content": msg, "is_error": True}
                )
                round_failed = True
                continue
            if not is_allowed_by_shell_allowlist(command, shell_allowlist):
                msg = f"Blocked: command is not in the shell allowlist ({', '.join(shell_allowlist)})."
                record_action(action_log_store, session_id, "run", tool.name, block["input"], "blocked", msg)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block["id"], "content": msg, "is_error": True}
                )
                round_failed = True
                continue
            # In dry-run mode the shell tool itself is a no-op (make_shell_tool's
            # dry_run branch), so there is nothing destructive to gate behind a
            # confirmation prompt -- skip it entirely rather than interrupting an
            # unattended preview run for a question with no real consequence either way.
            if not allow_shell and not dry_run:
                print(f"\n[confirm] run_shell({block['input']})")
                if not safe_confirm("Run this? [y/N] "):
                    record_action(
                        action_log_store, session_id, "run", tool.name, block["input"],
                        "declined", "User declined to run this tool.",
                    )
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
        record_action(
            action_log_store, session_id, "run", tool.name, block["input"],
            "error" if result.is_error else "ok", result.content,
        )
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
