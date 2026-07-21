import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from mazu.action_log.store import ActionLogStore, record_action
from mazu.agent.context import build_system_prompt
from mazu.llm.client import _split_model, run_turn, summarize_usage
from mazu.llm.errors import MazuAPIError
from mazu.llm.pricing import estimate_cost
from mazu.tools.registry import ToolRegistry
from mazu.usage.store import UsageStore

# Council members get read-only tools only. If they could write/edit/run shell, running
# several models against the same live project directory at once would let them clobber
# each other's changes — this mode is for getting independent opinions, not independent
# execution.
READ_ONLY_TOOL_NAMES = {"read_file", "list_dir", "glob_files", "recall", "list_skills"}

class _SharedCostTracker:
    """Thread-safe running total, shared across every council member's worker
    thread (see run_council's ThreadPoolExecutor) plus the main thread's lead-call
    check. Python's GIL does not make a read-then-compare-then-add sequence atomic,
    so concurrent members bumping the same total without a lock could lose updates
    (member A reads total=0.01, member B reads total=0.01, both add their own cost
    to that same stale value, one of the two additions is silently dropped) -- a
    real race, not a theoretical one, once members run truly in parallel.

    A `None` max_cost makes every check a permanent no-op (`is_exhausted` always
    False, `add_and_check` always returns False after adding), so callers never need
    a separate "if max_cost is set" branch anywhere else in this module.
    """

    def __init__(self, max_cost: float | None):
        self._max_cost = max_cost
        self._lock = threading.Lock()
        self._total = 0.0

    def add_and_check(self, cost: float | None) -> bool:
        """Adds cost (if trackable -- None means "couldn't estimate this call",
        same honest degradation `mazu run`'s cost_trackable gate already has) and
        returns whether the shared budget is now exhausted.
        """
        with self._lock:
            if cost is not None:
                self._total += cost
            return self._max_cost is not None and self._total >= self._max_cost

    def is_exhausted(self) -> bool:
        with self._lock:
            return self._max_cost is not None and self._total >= self._max_cost

    @property
    def total(self) -> float:
        with self._lock:
            return self._total


MEMBER_PROMPT_SUFFIX = """

You are one member of a council of AI models being asked the same question \
independently, without seeing the other members' answers. Give your best, concise \
recommendation. You may use the available read-only tools to inspect the project before \
answering, but you cannot write, edit, or run anything in this mode — this is an advisory \
round only, not execution."""

LEAD_PROMPT_TEMPLATE = """You are the lead reviewer synthesizing input from a council of \
{n} AI models who were each independently asked the following question:

{question}

Their responses:

{responses}

Compare them: note where they agree, where they genuinely disagree and why, and give a \
single final recommendation. Be decisive — the user needs one clear answer, not a summary \
of disagreement for its own sake."""


def _read_only_registry(full_registry: ToolRegistry) -> ToolRegistry:
    registry = ToolRegistry()
    for name in READ_ONLY_TOOL_NAMES:
        tool = full_registry.get(name)
        if tool is not None:
            registry.register(tool)
    return registry


def _ask_member(
    model: str,
    question: str,
    registry: ToolRegistry,
    system_prompt: str,
    max_rounds: int = 6,
    cost_tracker: "_SharedCostTracker | None" = None,
) -> tuple[str, list[dict], list[dict]]:
    """Returns (final_text, usage_list, action_records) -- one usage dict per model
    call this member made, and one action record dict per tool call. Deliberately
    returns both instead of logging them directly: this function runs inside a worker
    thread (see run_council's ThreadPoolExecutor), and both UsageStore's and
    ActionLogStore's sqlite3 connections are not safe to write from multiple threads.
    Logging happens back in the main thread once future.result() returns.
    """
    messages: list[dict] = [{"role": "user", "content": question}]
    member_system = system_prompt + MEMBER_PROMPT_SUFFIX
    final_text = "(no response)"
    usage_list: list[dict] = []
    action_records: list[dict] = []

    for _ in range(max_rounds):
        response = run_turn(messages, member_system, registry.schemas(), model=model)
        usage_list.append(response.usage)
        messages.append({"role": "assistant", "content": response.content})
        print(f"  [{model}] usage: {summarize_usage(response.usage)}")

        text_blocks = [b["text"] for b in response.content if b["type"] == "text"]
        if text_blocks:
            final_text = "\n".join(text_blocks)

        # Every round spends real money regardless of whether it ends the member's
        # turn or continues into tool use -- this must be tracked unconditionally,
        # BEFORE the stop_reason check below, or a member that answers directly
        # without ever using a tool (the common case for simple questions) would
        # never contribute its cost to the shared budget at all, silently letting
        # --max-cost do nothing. Only the DECISION to take another round is gated
        # by exhaustion; this round's own cost is already sunk by the time it's
        # known, matching mazu run's existing "the step that trips the cap still
        # completes" pattern.
        budget_exhausted = False
        if cost_tracker is not None:
            input_tokens = response.usage.get("input_tokens", response.usage.get("prompt_tokens", 0)) or 0
            output_tokens = response.usage.get("output_tokens", response.usage.get("completion_tokens", 0)) or 0
            round_cost = estimate_cost(model, input_tokens, output_tokens)
            budget_exhausted = cost_tracker.add_and_check(round_cost)

        if response.stop_reason != "tool_use":
            break

        if budget_exhausted:
            break

        tool_results = []
        for block in response.content:
            if block["type"] != "tool_use":
                continue
            tool = registry.get(block["name"])
            if tool is None:
                action_records.append(
                    {
                        "tool_name": block["name"],
                        "tool_input": block["input"],
                        "outcome": "unknown_tool",
                        "output_summary": f"Unknown tool: {block['name']}",
                    }
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
            result = tool.handler(block["input"])
            action_records.append(
                {
                    "tool_name": tool.name,
                    "tool_input": block["input"],
                    "outcome": "error" if result.is_error else "ok",
                    "output_summary": result.content,
                }
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

    return final_text, usage_list, action_records


def _log_usage(
    usage_store: UsageStore | None, session_id: str | None, model: str, usage: dict
) -> None:
    if usage_store is None:
        return
    provider_name, model_name = _split_model(model)
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    cost = estimate_cost(model, input_tokens, output_tokens)
    usage_store.log("council", session_id, provider_name, model_name, input_tokens, output_tokens, cost)


def run_council(
    question: str,
    models: list[str],
    lead_model: str,
    full_registry: ToolRegistry,
    memory_store=None,
    global_memory_store=None,
    skill_manager=None,
    usage_store: UsageStore | None = None,
    session_id: str | None = None,
    action_log_store: ActionLogStore | None = None,
    max_cost: float | None = None,
) -> str:
    system_prompt = build_system_prompt(
        memory_store, skill_manager, query=question, global_memory_store=global_memory_store
    )
    readonly_registry = _read_only_registry(full_registry)
    cost_tracker = _SharedCostTracker(max_cost)

    print(
        f"[cost] Council mode queries {len(models)} models independently, plus one lead "
        f"synthesis call — expect roughly {len(models)}x the cost of a single `mazu run`/"
        "`mazu chat` call (more if members use multiple rounds of tool calls).\n"
    )
    print(
        f"Asking {len(models)} models independently and in parallel "
        f"(read-only, advisory round): {', '.join(models)}\n"
        f"Lead (synthesizes the final answer): {lead_model}\n"
    )
    responses: dict[str, str] = {}
    # Members are independent (no shared mutable state — each gets its own message
    # list; `readonly_registry` is only ever read from, never written to, so sharing
    # it across threads is safe), so running them concurrently cuts wall-clock time
    # roughly to the slowest single member instead of the sum of all of them.
    # `cost_tracker` IS shared mutable state across members, by design (item 3) --
    # it's internally lock-protected specifically so this concurrent sharing is safe.
    with ThreadPoolExecutor(max_workers=max(len(models), 1)) as executor:
        future_to_model = {
            executor.submit(
                _ask_member, model, question, readonly_registry, system_prompt,
                cost_tracker=cost_tracker,
            ): model
            for model in models
        }
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                text, usage_list, action_records = future.result()
                responses[model] = text
                # Logged here, in the main thread, not inside the worker -- see
                # _ask_member's docstring for why.
                for usage in usage_list:
                    _log_usage(usage_store, session_id, model, usage)
                for record in action_records:
                    record_action(
                        action_log_store, session_id, "council",
                        record["tool_name"], record["tool_input"],
                        record["outcome"], record["output_summary"],
                    )
            except Exception as e:
                responses[model] = f"(failed: {e})"
            print(f"[{model}] done")
    print()

    if cost_tracker.is_exhausted():
        final_text = (
            f"(Skipped lead synthesis: the --max-cost budget (${max_cost:.2f}) was reached "
            f"during the member round(s), at an estimated ${cost_tracker.total:.4f}. "
            "Individual member answers above are still available, but no synthesized "
            "recommendation could be produced.)"
        )
        print(f"[cost] {final_text}\n")
        return final_text

    responses_text = "\n\n".join(f"=== {model} ===\n{answer}" for model, answer in responses.items())
    lead_prompt = LEAD_PROMPT_TEMPLATE.format(
        n=len(models), question=question, responses=responses_text
    )
    try:
        lead_response = run_turn(
            [{"role": "user", "content": lead_prompt}], system_prompt, [], model=lead_model
        )
    except MazuAPIError as e:
        final_text = (
            f"(Lead model '{lead_model}' failed: {e}. Individual member answers above are "
            "still available, but no synthesized recommendation could be produced.)"
        )
        print(f"[error] {final_text}\n")
        return final_text

    final_text = "\n".join(b["text"] for b in lead_response.content if b["type"] == "text")
    _log_usage(usage_store, session_id, lead_model, lead_response.usage)

    print(f"  [{lead_model}] usage: {summarize_usage(lead_response.usage)}")
    print(f"=== Lead ({lead_model}) synthesis ===\n{final_text}\n")
    return final_text
