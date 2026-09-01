from pathlib import Path

from mazu.action_log.store import ActionLogStore, record_action
from mazu.agent.compaction import (
    DEFAULT_KEEP_RECENT,
    compact_if_needed,
    compact_messages,
    extract_memories_on_compaction,
    force_compact,
    summarize_for_compaction,
)
from mazu.agent.context import build_system_prompt
from mazu.agent.interaction import safe_confirm
from mazu.agent.session import finalize_session
from mazu.banner import print_banner
from mazu.chat.store import ChatStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.config import router_suggestions_enabled
from mazu.llm.client import _split_model, default_model, extract_cost_tokens, run_turn_stream, summarize_usage
from mazu.llm.errors import MazuAPIError, MazuContextLengthError
from mazu.llm.pricing import estimate_cost
from mazu.memory.belief_log import BeliefShadowLogStore
from mazu.memory.store import MemoryStore
from mazu.runs.router import suggest_model
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.tools.registry import ToolRegistry
from mazu.tools.shell import denylist_reason, is_allowed_by_shell_allowlist
from mazu.usage.store import UsageStore


def _print_chat_router_suggestion(checkpoint_manager: CheckpointManager | None, task: str) -> None:
    """Same passive, best-effort suggestion as mazu run's (see
    mazu/cli.py::_print_router_suggestion and mazu/runs/router.py's module
    docstring) -- `mazu chat` has no task text at session start, unlike `mazu run`,
    so this is deferred to the first real user message instead. Uses
    checkpoint_manager.root rather than taking a new `root` param, since chat's one
    call site (mazu/cli.py::chat) always constructs a real CheckpointManager.
    """
    if checkpoint_manager is None or not router_suggestions_enabled():
        return
    try:
        root = checkpoint_manager.root
        run_store = RunStore(root / ".mazu" / "runs.db")
        usage_store = UsageStore(Path.home() / ".mazu" / "usage.db")
        try:
            suggestion = suggest_model(run_store, usage_store, task)
        finally:
            run_store.close()
            usage_store.close()
        if suggestion:
            print(suggestion)
    except Exception:
        pass


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
    usage_store: UsageStore | None = None,
    action_log_store: ActionLogStore | None = None,
    shell_allowlist: list[str] | None = None,
    chat_store: ChatStore | None = None,
    resumed_messages: list[dict] | None = None,
    belief_log: BeliefShadowLogStore | None = None,
    auto_approve: bool = False,
) -> None:
    messages: list[dict] = list(resumed_messages) if resumed_messages else []
    # Left None even when resuming: the first new user input still needs its own
    # system_prompt built (a fresh memory-retrieval query and, if unset, a router
    # suggestion) exactly like a brand-new session's first message -- resuming
    # only pre-populates prior conversation turns, not this per-process state.
    system_prompt = None
    total_cost = 0.0
    # A resumed chat reuses the SAME session_id (and therefore the same `sessions`
    # row MemoryStore already has for it, from the original process) --
    # start_session()'s INSERT would violate that row's primary key and crash.
    # Identical fix to mazu/agent/autonomous.py's run_autonomous() for `mazu run
    # --resume`; found live testing `mazu chat --resume` the same way that one was.
    is_resume = resumed_messages is not None
    # A single-element list, not a plain bool, so /auto can flip it from inside the
    # input loop below and have _run_until_done (called after the toggle) see the
    # new value -- a bare local `auto_approve` reassignment wouldn't be visible to
    # the closure-free call below without threading a return value back out.
    auto_approve_flag = [auto_approve]

    print_banner()
    provider_name, model_name = _split_model(model or default_model())
    resolved_model = f"{provider_name}:{model_name}"
    cost_trackable = estimate_cost(resolved_model, 0, 0) is not None
    print(f"model: {resolved_model}")
    print(
        "chat — type your task. Commands: /checkpoint, /rollback [id], /compact, /auto. Ctrl+C to quit.\n"
    )
    if auto_approve_flag[0]:
        print(
            "[auto] auto-approve is ON for this session (--auto-approve) -- destructive "
            "tools (shell, file delete, skill delete) will run without asking. Type /auto "
            "to turn it off.\n"
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
                _handle_checkpoint(checkpoint_manager, messages, session_id)
                continue

            if user_input == "/rollback" or user_input.startswith("/rollback "):
                new_messages = _handle_rollback(checkpoint_manager, user_input, messages)
                if new_messages is not None:
                    messages[:] = new_messages
                continue

            if user_input == "/compact":
                _handle_compact(messages, model, memory_store, usage_store, session_id, belief_log)
                continue

            if user_input == "/auto":
                auto_approve_flag[0] = not auto_approve_flag[0]
                if auto_approve_flag[0]:
                    print(
                        "[auto] auto-approve is now ON -- destructive tools (shell, file "
                        "delete, skill delete) will run without asking. Type /auto again to "
                        "turn it back off."
                    )
                else:
                    print("[auto] auto-approve is now OFF -- destructive tools will ask for confirmation again.")
                continue

            if system_prompt is None:
                if memory_store is not None and not is_resume:
                    memory_store.start_session(session_id)
                system_prompt = build_system_prompt(
                    memory_store, skill_manager, query=user_input, global_memory_store=global_memory_store
                )
                if system_prompt.strip() != "":
                    print("[memory] loaded prior context relevant to this task\n")
                if model is None:
                    _print_chat_router_suggestion(checkpoint_manager, user_input)

            # Snapshotted BEFORE appending this turn's user message -- see
            # _run_until_done's own comment on messages_before_turn for why: an
            # unrecoverable MazuAPIError used to leave the failed message stuck in
            # `messages` forever, so every later turn kept resending it and
            # re-triggering the exact same error.
            messages_before_turn = list(messages)
            messages.append({"role": "user", "content": user_input})
            if chat_store is not None:
                chat_store.append_message(session_id, "user", user_input, model=resolved_model)
            total_cost = _run_until_done(
                messages,
                messages_before_turn,
                registry,
                system_prompt,
                model,
                resolved_model,
                cost_trackable,
                usage_store,
                session_id,
                total_cost,
                action_log_store,
                shell_allowlist,
                chat_store,
                memory_store,
                belief_log,
                auto_approve_flag[0],
            )
    finally:
        if memory_store is not None:
            finalize_session(
                memory_store, session_id, messages, model=model, usage_store=usage_store,
                belief_log=belief_log,
            )
        if global_memory_store is not None:
            global_memory_store.close()
        if usage_store is not None:
            usage_store.close()
        if action_log_store is not None:
            action_log_store.close()
        if chat_store is not None:
            chat_store.close()
        if belief_log is not None:
            belief_log.close()


def _handle_checkpoint(
    checkpoint_manager: CheckpointManager | None, messages: list[dict], session_id: str | None = None
) -> None:
    if checkpoint_manager is None:
        print("Checkpointing is not available.")
        return
    entry = checkpoint_manager.snapshot(messages, trigger="manual", session_id=session_id)
    print(f"[checkpoint] {entry['id']} saved (commit {entry['git_commit'][:8]})")


def _handle_compact(
    messages: list[dict],
    model: str | None,
    memory_store: MemoryStore | None,
    usage_store: UsageStore | None,
    session_id: str,
    belief_log: BeliefShadowLogStore | None = None,
) -> None:
    if len(messages) <= DEFAULT_KEEP_RECENT + 1:
        print("[compact] nothing to compact yet — conversation is already short.")
        return
    before = len(messages)
    on_compacted = extract_memories_on_compaction(
        memory_store, session_id, model, usage_store=usage_store, belief_log=belief_log
    )
    new_messages = compact_messages(
        messages,
        lambda mm: summarize_for_compaction(mm, model, usage_store=usage_store, session_id=session_id),
        keep_recent=DEFAULT_KEEP_RECENT,
        on_compacted=on_compacted,
    )
    if new_messages is messages:
        print("[compact] nothing safe to cut yet (all messages are within the recent window).")
        return
    messages[:] = new_messages
    print(
        f"[compact] compacted {before} messages down to {len(messages)} in this turn's context. "
        f"The full original conversation is still saved — see `mazu chat --list` / `--resume`."
    )


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
    messages: list[dict],
    messages_before_turn: list[dict],
    registry: ToolRegistry,
    system_prompt: str,
    model: str | None,
    resolved_model: str,
    cost_trackable: bool,
    usage_store: UsageStore | None,
    session_id: str,
    total_cost: float,
    action_log_store: ActionLogStore | None = None,
    shell_allowlist: list[str] | None = None,
    chat_store: ChatStore | None = None,
    memory_store: MemoryStore | None = None,
    belief_log: BeliefShadowLogStore | None = None,
    auto_approve: bool = False,
) -> float:
    """`messages_before_turn` is a snapshot of `messages` taken by the caller right
    before this turn's user message was appended -- real bug found via live
    testing (mazu-web's chat, which mirrors this function closely): a turn that
    fails with a non-retryable MazuAPIError (e.g. an unsupported request that the
    model rejects outright) used to leave that failed user message sitting in
    `messages` forever. Since every LLM call resends the full history, every
    subsequent turn kept resending the bad message and re-triggering the exact
    same error, permanently breaking the session. Restoring this snapshot on
    failure below discards the whole failed turn (the user message, and any
    compaction or partial tool rounds attempted within it) so the next turn
    starts clean. `chat_store`'s durable record is deliberately NOT rolled back --
    it keeps everything actually attempted, same as it keeps messages compaction
    later summarizes away.
    """
    provider_name, model_name = _split_model(resolved_model)
    while True:
        on_compacted = extract_memories_on_compaction(
            memory_store, session_id, model, usage_store=usage_store, belief_log=belief_log
        )
        new_messages, compacted = compact_if_needed(
            messages, model, usage_store=usage_store, session_id=session_id, on_compacted=on_compacted
        )
        if compacted:
            messages[:] = new_messages  # mutate in place -- messages is passed in by reference
            print(
                f"[context] compacted conversation history to stay within budget "
                f"({len(messages)} messages remain in this turn's context). Nothing is "
                f"lost — the full conversation is still saved (`mazu chat --list` / `--resume`)."
            )

        streamed_any_text = False

        def _on_delta(chunk: str) -> None:
            nonlocal streamed_any_text
            streamed_any_text = True
            print(chunk, end="", flush=True)

        try:
            response = run_turn_stream(
                messages, system_prompt, registry.schemas(), on_delta=_on_delta, model=model
            )
        except MazuContextLengthError:
            print(
                "\n[context] hit the model's context limit — compacting aggressively and "
                "retrying once. The full conversation remains saved; only what's sent to "
                "the model just shrank."
            )
            on_compacted = extract_memories_on_compaction(
                memory_store, session_id, model, usage_store=usage_store, belief_log=belief_log
            )
            messages[:] = force_compact(
                messages, model, usage_store=usage_store, session_id=session_id, on_compacted=on_compacted
            )
            try:
                response = run_turn_stream(
                    messages, system_prompt, registry.schemas(), on_delta=_on_delta, model=model
                )
            except MazuAPIError as e:
                messages[:] = messages_before_turn
                print(f"\n[error] {e}\nReturning to the prompt — try again, or use /rollback.")
                return total_cost
        except MazuAPIError as e:
            messages[:] = messages_before_turn
            print(f"\n[error] {e}\nReturning to the prompt — try again, or use /rollback.")
            return total_cost
        messages.append({"role": "assistant", "content": response.content})
        if chat_store is not None:
            chat_store.append_message(session_id, "assistant", response.content, model=resolved_model)
        # Text was already printed live as it streamed in; usage can only be known
        # once the stream is fully done, so it prints after (not before, like the
        # old non-streaming order) with a newline to close off the streamed line.
        if streamed_any_text:
            print()

        usage = response.usage
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        cost_input_tokens, _, cache_read_tokens, cache_write_tokens = extract_cost_tokens(usage)
        step_cost = (
            estimate_cost(
                resolved_model, cost_input_tokens, output_tokens,
                cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
            )
            if cost_trackable else None
        )
        cost_suffix = ""
        if step_cost is not None:
            total_cost += step_cost
            cost_suffix = f" | ~${total_cost:.4f} total"
        if usage_store is not None:
            usage_store.log(
                "chat", session_id, provider_name, model_name, input_tokens, output_tokens, step_cost,
                cache_read_tokens=cache_read_tokens or None, cache_write_tokens=cache_write_tokens or None,
            )
        print(f"[usage] {summarize_usage(usage)}{cost_suffix}")

        if response.stop_reason != "tool_use":
            return total_cost

        tool_results = []
        for block in response.content:
            if block["type"] != "tool_use":
                continue
            tool = registry.get(block["name"])
            if tool is None:
                record_action(
                    action_log_store, session_id, "chat", block["name"], block["input"],
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
                continue
            if tool.name == "run_shell":
                command = block["input"].get("command", "")
                reason = denylist_reason(command)
                if reason is not None:
                    msg = f"Blocked: command {reason} (safety denylist)."
                    record_action(action_log_store, session_id, "chat", tool.name, block["input"], "blocked", msg)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block["id"], "content": msg, "is_error": True}
                    )
                    continue
                if not is_allowed_by_shell_allowlist(command, shell_allowlist):
                    msg = f"Blocked: command is not in the shell allowlist ({', '.join(shell_allowlist)})."
                    record_action(action_log_store, session_id, "chat", tool.name, block["input"], "blocked", msg)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block["id"], "content": msg, "is_error": True}
                    )
                    continue
            if tool.destructive:
                if auto_approve:
                    print(f"\n[auto] running {tool.name}({block['input']}) without confirmation (auto mode is on)")
                elif not _confirm(tool.name, block["input"]):
                    record_action(
                        action_log_store, session_id, "chat", tool.name, block["input"],
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
            record_action(
                action_log_store, session_id, "chat", tool.name, block["input"],
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
        messages.append({"role": "user", "content": tool_results})
        if chat_store is not None:
            chat_store.append_message(session_id, "user", tool_results, model=resolved_model)
