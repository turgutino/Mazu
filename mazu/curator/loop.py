from dataclasses import dataclass, field

from mazu.curator.budget import CuratorBudget
from mazu.curator.client import run_curator_turn
from mazu.llm.client import summarize_usage
from mazu.llm.pricing import estimate_cost
from mazu.tools.registry import ToolRegistry


@dataclass
class CuratorPassResult:
    final_text: str
    usage_list: list[dict] = field(default_factory=list)
    stop_reason: str = "end_turn"
    total_cost: float = 0.0


def run_curator_pass(
    task_prompt: str,
    system_prompt: str,
    registry: ToolRegistry,
    max_rounds: int = 8,
    budget: CuratorBudget | None = None,
    verbose: bool = False,
) -> CuratorPassResult:
    """Modeled directly on mazu.agent.council._ask_member's shape -- fixed small
    round cap, plain for loop, no CheckpointManager, no shell tool, no
    dirty-working-tree gate, silent unless verbose. Unlike _ask_member, this
    doesn't return raw action_records for the caller to log separately -- every
    Curator tool handler already writes its own curator_log entry directly (see
    mazu/curator/tools/*.py), since there's no multi-threading concern here that
    would make that unsafe (Curator runs one area at a time, single-threaded).
    """
    messages: list[dict] = [{"role": "user", "content": task_prompt}]
    final_text = "(no response)"
    usage_list: list[dict] = []
    stop_reason = "max_rounds"

    for _ in range(max_rounds):
        response = run_curator_turn(messages, system_prompt, registry.schemas())
        usage_list.append(response.usage)
        messages.append({"role": "assistant", "content": response.content})
        if verbose:
            print(f"  [curator] usage: {summarize_usage(response.usage)}")

        text_blocks = [b["text"] for b in response.content if b["type"] == "text"]
        if text_blocks:
            final_text = "\n".join(text_blocks)

        budget_exhausted = False
        if budget is not None:
            input_tokens = response.usage.get("input_tokens", response.usage.get("prompt_tokens", 0)) or 0
            output_tokens = response.usage.get("output_tokens", response.usage.get("completion_tokens", 0)) or 0
            round_cost = estimate_cost(_current_model_label(), input_tokens, output_tokens)
            budget_exhausted = budget.add_and_check(round_cost)

        if response.stop_reason != "tool_use":
            stop_reason = "end_turn"
            break
        if budget_exhausted:
            stop_reason = "budget"
            break

        tool_results = []
        for block in response.content:
            if block["type"] != "tool_use":
                continue
            tool = registry.get(block["name"])
            if tool is None:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block["id"],
                    "content": f"Unknown tool: {block['name']}", "is_error": True,
                })
                continue
            result = tool.handler(block["input"])
            if verbose:
                print(f"  [curator] {block['name']}({block['input']}) -> {result.content[:200]}")
            tool_results.append({
                "type": "tool_result", "tool_use_id": block["id"],
                "content": result.content, "is_error": result.is_error,
            })
        messages.append({"role": "user", "content": tool_results})
    else:
        stop_reason = "max_rounds"

    return CuratorPassResult(
        final_text=final_text, usage_list=usage_list, stop_reason=stop_reason,
        total_cost=budget.total if budget is not None else 0.0,
    )


def _current_model_label() -> str:
    """Curator's own resolved model string ('provider:model'), used only for cost
    estimation (llm.pricing.estimate_cost keys off the same string format every
    other cost-tracking call site in Mazu already uses)."""
    from mazu.curator.config import curator_model

    model = curator_model()
    assert model is not None, "run_curator_pass called with Curator unconfigured"
    return model
