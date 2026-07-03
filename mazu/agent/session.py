from mazu.memory.extraction import extract_memories
from mazu.memory.store import MemoryStore


def finalize_session(
    memory_store: MemoryStore, session_id: str, messages: list[dict], model: str | None = None
) -> None:
    """Runs once at the end of a session: the cheap-model safety net that catches
    whatever the agent forgot to `remember` explicitly during the conversation.
    `model` should be the model the session actually used, so extraction stays on the
    same provider/API key rather than assuming Anthropic.
    """
    try:
        extracted = extract_memories(messages, model=model)
    except Exception as e:
        print(f"[memory] extraction skipped: {e}")
        extracted = []

    inserted = 0
    for item in extracted:
        try:
            # This is a redundant safety net on top of the explicit `remember` tool, not
            # the primary write path — skip anything that's already stored under the same
            # (or an obvious rephrasing of the same) title/category so it doesn't pile up
            # duplicates every session.
            if (
                memory_store.find_duplicate(item["category"], item["title"], item["body"])
                is not None
            ):
                continue
            memory_store.add(
                category=item["category"],
                title=item["title"],
                body=item["body"],
                tags=item.get("tags", ""),
                source="auto_extracted",
                session_id=session_id,
            )
            inserted += 1
        except Exception:
            continue

    if inserted:
        print(f"[memory] saved {inserted} new memories from this session")

    summary = extracted[0]["title"] if extracted else ""
    memory_store.end_session(session_id, task_summary=summary)
    memory_store.close()
