from concurrent.futures import ThreadPoolExecutor, as_completed

from mazu.agent.context import build_system_prompt
from mazu.llm.client import run_turn, summarize_usage
from mazu.llm.errors import MazuAPIError
from mazu.tools.registry import ToolRegistry

# Council members get read-only tools only. If they could write/edit/run shell, running
# several models against the same live project directory at once would let them clobber
# each other's changes — this mode is for getting independent opinions, not independent
# execution.
READ_ONLY_TOOL_NAMES = {"read_file", "list_dir", "glob_files", "recall", "list_skills"}

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
    model: str, question: str, registry: ToolRegistry, system_prompt: str, max_rounds: int = 6
) -> str:
    messages: list[dict] = [{"role": "user", "content": question}]
    member_system = system_prompt + MEMBER_PROMPT_SUFFIX
    final_text = "(no response)"

    for _ in range(max_rounds):
        response = run_turn(messages, member_system, registry.schemas(), model=model)
        messages.append({"role": "assistant", "content": response.content})
        print(f"  [{model}] usage: {summarize_usage(response.usage)}")

        text_blocks = [b["text"] for b in response.content if b["type"] == "text"]
        if text_blocks:
            final_text = "\n".join(text_blocks)

        if response.stop_reason != "tool_use":
            break

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

    return final_text


def run_council(
    question: str,
    models: list[str],
    lead_model: str,
    full_registry: ToolRegistry,
    memory_store=None,
    global_memory_store=None,
    skill_manager=None,
) -> str:
    system_prompt = build_system_prompt(
        memory_store, skill_manager, query=question, global_memory_store=global_memory_store
    )
    readonly_registry = _read_only_registry(full_registry)

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
    with ThreadPoolExecutor(max_workers=max(len(models), 1)) as executor:
        future_to_model = {
            executor.submit(_ask_member, model, question, readonly_registry, system_prompt): model
            for model in models
        }
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                responses[model] = future.result()
            except Exception as e:
                responses[model] = f"(failed: {e})"
            print(f"[{model}] done")
    print()

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

    print(f"  [{lead_model}] usage: {summarize_usage(lead_response.usage)}")
    print(f"=== Lead ({lead_model}) synthesis ===\n{final_text}\n")
    return final_text
