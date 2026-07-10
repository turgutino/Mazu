from typing import Callable

from mazu.llm.client import _split_model, default_model, run_turn
from mazu.memory.extraction import CHEAP_MODEL_BY_PROVIDER, render_transcript

# Rough token estimate: ~4 characters per token is a standard, conservative rule of
# thumb for English text across tokenizers. Used only to decide *when* to compact,
# not for billing -- same "approximate, documented, not exact" spirit already used
# for cost estimation in llm/pricing.py.
CHARS_PER_TOKEN_ESTIMATE = 4

# Deliberately conservative and shared across providers rather than looked up per
# model: better to compact a bit earlier than strictly necessary on a large-context
# model than to risk a context-length error on a smaller one. Revisit with a real
# per-model context-window table if this proves too aggressive in practice.
DEFAULT_TRIGGER_TOKENS = 60_000
DEFAULT_KEEP_RECENT = 10
# Used only for the reactive fallback after an actual MazuContextLengthError -- keep
# much less so the retry has the best chance of fitting.
AGGRESSIVE_KEEP_RECENT = 4

COMPACTION_SUMMARY_PREFIX = (
    "## Summary of earlier conversation (compacted to stay within the context budget)\n\n"
)
COMPACTION_ACK_TEXT = "Understood — continuing from that summary."

COMPACTION_SYSTEM = (
    "You summarize the earlier portion of a coding agent's conversation so it can "
    "continue with less context. Preserve concrete facts: what the original task was, "
    "what has been done so far, what files were touched, what decisions were made, and "
    "anything still outstanding. Omit routine back-and-forth that doesn't affect what "
    "happens next. Be concise, but do not drop information the agent will still need."
)


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "tool_use":
            parts.append(str(block.get("input", "")))
        elif block.get("type") == "tool_result":
            parts.append(str(block.get("content", "")))
    return " ".join(parts)


def estimate_tokens(messages: list[dict]) -> int:
    total_chars = sum(len(_message_text(m["content"])) for m in messages)
    return total_chars // CHARS_PER_TOKEN_ESTIMATE


def needs_compaction(messages: list[dict], trigger_tokens: int = DEFAULT_TRIGGER_TOKENS) -> bool:
    return estimate_tokens(messages) > trigger_tokens


def _round_boundaries(messages: list[dict]) -> list[int]:
    """Indices at which it's safe to cut the message list: every position that does
    NOT fall between an assistant tool_use turn and its paired tool_result turn.
    Cutting inside that gap would separate a tool_use block from the tool_result the
    API requires immediately after it, producing an invalid request.
    """
    boundaries = [0]
    pending_tool_use = False
    for i, msg in enumerate(messages):
        content = msg["content"]
        if msg["role"] == "assistant":
            pending_tool_use = isinstance(content, list) and any(
                b.get("type") == "tool_use" for b in content
            )
        else:
            pending_tool_use = False
        if not pending_tool_use:
            boundaries.append(i + 1)
    return boundaries


def compact_messages(
    messages: list[dict],
    summarize_fn: Callable[[list[dict]], str],
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> list[dict]:
    """Replaces the oldest portion of `messages` with a two-message synthetic
    exchange (a user-role summary + an assistant-role acknowledgment) covering
    everything up to a safe cut point, keeping the most recent `keep_recent`
    messages verbatim.

    Correctness constraints, both required by every provider's Messages/Chat API:
    1. Never cut between an assistant's tool_use block and its paired tool_result
       (see _round_boundaries) — a long run of consecutive tool-call rounds (the most
       common shape for a real `mazu run` session) means most safe boundaries land
       right before another *assistant* message, not a user one, so the synthetic
       replacement's shape adapts to whatever role follows the cut (see below) rather
       than restricting which boundaries are usable in the first place.
    2. The synthetic replacement must always start with role="user" (matching what
       messages[0] always was) and must end in the *opposite* role from whatever
       message immediately follows the cut, so the full list keeps strictly
       alternating user/assistant with no gap or clash at the seam.

    Returns `messages` unchanged (the same object, not a copy — callers can check
    `result is messages` to detect a no-op) if there's nothing safe/worthwhile to
    compact.
    """
    if len(messages) <= keep_recent + 1:
        return messages

    boundaries = _round_boundaries(messages)
    target = len(messages) - keep_recent
    valid = [b for b in boundaries if 1 < b <= target]
    if not valid:
        return messages
    cut = max(valid)

    summary_text = summarize_fn(messages[:cut])
    tail = messages[cut:]
    synthetic = [{"role": "user", "content": COMPACTION_SUMMARY_PREFIX + summary_text}]
    if not tail or tail[0]["role"] == "user":
        # tail starts with "user" (or is empty) -- append an assistant ack so the
        # synthetic block itself ends in "assistant", keeping the seam alternating.
        synthetic.append({"role": "assistant", "content": COMPACTION_ACK_TEXT})
    return [*synthetic, *tail]


def _compaction_model(main_model: str | None) -> str:
    # Summarizing is a cheap, structured task -- reuse the same cheap-tier,
    # same-provider model choice memory/extraction.py already established, so this
    # never silently requires a second provider's API key.
    provider_name, _ = _split_model(main_model or default_model())
    return CHEAP_MODEL_BY_PROVIDER.get(provider_name, CHEAP_MODEL_BY_PROVIDER["anthropic"])


def summarize_for_compaction(messages_to_summarize: list[dict], model: str | None) -> str:
    transcript = render_transcript(messages_to_summarize)
    response = run_turn(
        messages=[{"role": "user", "content": f"Summarize this:\n\n{transcript}"}],
        system=COMPACTION_SYSTEM,
        tools=[],
        model=_compaction_model(model),
    )
    text_blocks = [b["text"] for b in response.content if b.get("type") == "text"]
    return "\n".join(text_blocks) or "(no summary content returned)"


def compact_if_needed(
    messages: list[dict],
    model: str | None,
    trigger_tokens: int = DEFAULT_TRIGGER_TOKENS,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> tuple[list[dict], bool]:
    """Proactive check, meant to be called once per round before the main model
    call. Returns (new_messages, True) if the estimated size exceeded
    `trigger_tokens` and compaction actually happened; otherwise (messages, False)
    with `messages` returned unchanged.
    """
    if not needs_compaction(messages, trigger_tokens):
        return messages, False
    compacted = compact_messages(
        messages, lambda mm: summarize_for_compaction(mm, model), keep_recent=keep_recent
    )
    return compacted, compacted is not messages


def force_compact(
    messages: list[dict], model: str | None, keep_recent: int = AGGRESSIVE_KEEP_RECENT
) -> list[dict]:
    """Reactive fallback for when a MazuContextLengthError already happened despite
    the proactive check (e.g. the char-based estimate undershot, or a single message
    was already huge). Skips the trigger_tokens gate entirely since the error is
    itself proof compaction is needed, and keeps much less than the proactive default
    so the retry has the best realistic chance of fitting.
    """
    return compact_messages(
        messages, lambda mm: summarize_for_compaction(mm, model), keep_recent=keep_recent
    )
