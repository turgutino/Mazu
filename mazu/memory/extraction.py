from mazu.llm.client import _split_model, default_model, run_forced_tool

# Cheap-tier model per provider, used only for this end-of-session extraction pass —
# not the model the user picked for the main conversation. Matches the provider of
# whatever model the session actually used, so a DeepSeek-only or OpenAI-only setup
# never needs an unrelated provider's key just to get memory extraction.
CHEAP_MODEL_BY_PROVIDER = {
    "anthropic": "anthropic:claude-haiku-4-5",
    "openai": "openai:gpt-5-mini",
    "deepseek": "deepseek:deepseek-chat",
}

EXTRACTION_SYSTEM = "You extract durable, reusable memories from coding session transcripts."

EXTRACTION_INSTRUCTIONS = """Given the following coding session transcript, extract 0-5 \
durable memories worth persisting for future sessions on this project: architectural \
decisions made, conventions established, mistakes made and how they were fixed, and the \
overall task outcome. Only extract things genuinely worth remembering long-term — skip \
trivial back-and-forth.

Do NOT extract personal facts about the person themselves (their name, age, preferred \
language, experience level, or similar) — those belong in a separate, global store that \
this project-scoped pass does not write to, and get saved through a different path when the \
person explicitly states them. If the transcript is purely about the person (not the \
project or its code), extract nothing from it here.

Be conservative: if the session was purely a question that got answered (nothing new was \
decided, built, fixed, or changed), return an empty list rather than restating the answer as \
a "new" memory — you cannot see what's already stored, so when in doubt about whether \
something is already established, still extract it (a downstream duplicate check will catch \
exact repeats), but do not manufacture a memory out of a session that added no new \
information at all.

Call record_memories with your findings (an empty list is fine if nothing is worth keeping)."""

RECORD_MEMORIES_TOOL = {
    "name": "record_memories",
    "description": "Record the durable memories extracted from this session.",
    "input_schema": {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["decision", "convention", "mistake", "task_outcome", "fact"],
                        },
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "tags": {"type": "string"},
                    },
                    "required": ["category", "title", "body"],
                },
            }
        },
        "required": ["memories"],
    },
}


def render_transcript(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
            continue
        for block in content:
            if block.get("type") == "text":
                lines.append(f"{role}: {block['text']}")
            elif block.get("type") == "tool_use":
                lines.append(f"{role} (tool call): {block['name']}({block['input']})")
            elif block.get("type") == "tool_result":
                snippet = str(block.get("content", ""))[:300]
                lines.append(f"tool_result: {snippet}")
    return "\n".join(lines)


def _extraction_model(main_model: str | None) -> str:
    provider_name, _ = _split_model(main_model or default_model())
    return CHEAP_MODEL_BY_PROVIDER.get(provider_name, CHEAP_MODEL_BY_PROVIDER["anthropic"])


def extract_memories(messages: list[dict], model: str | None = None) -> list[dict]:
    """`model` should be whatever model the main session used, so extraction stays on
    the same provider (and thus the same already-configured API key) rather than
    silently requiring Anthropic regardless of what the user is actually using.
    """
    transcript = render_transcript(messages)
    if not transcript.strip():
        return []
    result = run_forced_tool(
        messages=[
            {
                "role": "user",
                "content": f"{EXTRACTION_INSTRUCTIONS}\n\nTranscript:\n{transcript}",
            }
        ],
        system=EXTRACTION_SYSTEM,
        tool=RECORD_MEMORIES_TOOL,
        model=_extraction_model(model),
    )
    return result.get("memories", [])
