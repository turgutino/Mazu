from mazu.llm.client import _split_model, extract_cost_tokens
from mazu.llm.pricing import estimate_cost
from mazu.memory.belief import run_shadow_belief_batch
from mazu.memory.belief_log import BeliefShadowLogStore
from mazu.memory.extraction import extract_memories, extraction_model
from mazu.memory.store import MemoryStore
from mazu.usage.store import UsageStore


def insert_extracted_memories(
    memory_store: MemoryStore,
    session_id: str,
    messages: list[dict],
    model: str | None = None,
    usage_store: UsageStore | None = None,
    belief_log: BeliefShadowLogStore | None = None,
) -> tuple[int, list[dict]]:
    """Extracts durable memories from `messages` and inserts the non-duplicate ones
    into `memory_store`. Safe to call more than once against the same session_id --
    dedup is content-keyed via find_duplicate, not session-keyed -- which is what
    lets chat compaction call this mid-conversation on each compacted-away chunk,
    not just once at session end like `finalize_session` does.

    `model` should be the model the session actually used, so extraction stays on the
    same provider/API key rather than assuming Anthropic. Returns (inserted count,
    raw extracted list) -- callers that also need the extracted items (e.g.
    finalize_session's end-of-session summary) don't have to re-extract.

    `belief_log`, when given, additionally runs the SHADOW-MODE belief-maintenance
    check (mazu.memory.belief.run_shadow_belief_batch) against the extracted items
    before the existing dedup/insert loop below -- purely observational, logs what
    action a live contradiction-detection system WOULD take, never mutates
    `memory_store`. Defaults to None (opt-in): omitting it reproduces today's
    exact behavior, with zero added cost or risk, for every existing caller.
    """
    try:
        extracted, usage = extract_memories(messages, model=model)
    except Exception as e:
        print(f"[memory] extraction skipped: {e}")
        return 0, []

    if belief_log is not None and extracted:
        try:
            run_shadow_belief_batch(
                memory_store, belief_log, session_id, extracted, model=model, usage_store=usage_store
            )
        except Exception as e:
            # Shadow-mode evidence collection must never be able to affect the
            # real insert path below -- a failure here is swallowed, same
            # "best-effort, never crashes the session" contract as extraction itself.
            print(f"[memory] belief shadow-check skipped: {e}")

    if usage_store is not None and usage:
        provider_name, model_name = _split_model(extraction_model(model))
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        cost_input_tokens, _, cache_read_tokens, cache_write_tokens = extract_cost_tokens(usage)
        cost = estimate_cost(
            f"{provider_name}:{model_name}", cost_input_tokens, output_tokens,
            cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
        )
        usage_store.log(
            "memory_extraction", session_id, provider_name, model_name, input_tokens, output_tokens, cost,
            cache_read_tokens=cache_read_tokens or None, cache_write_tokens=cache_write_tokens or None,
        )

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

    return inserted, extracted


def finalize_session(
    memory_store: MemoryStore,
    session_id: str,
    messages: list[dict],
    model: str | None = None,
    usage_store: UsageStore | None = None,
    belief_log: BeliefShadowLogStore | None = None,
    dry_run: bool = False,
) -> int:
    """Runs once at the end of a session: the cheap-model safety net that catches
    whatever the agent forgot to `remember` explicitly during the conversation.
    Returns the number of memories actually inserted (post-dedup), for callers
    building an end-of-run report.

    `belief_log`, when given, is forwarded to insert_extracted_memories() to run
    the SHADOW-MODE belief check (see mazu.memory.belief) -- purely observational,
    never mutates memory_store. Defaults to None: omitting it reproduces today's
    exact behavior for every existing caller.

    `dry_run`, when True, skips extraction entirely -- real bug found live:
    `mazu run --dry-run` used to call this unconditionally, which made a real,
    billed LLM call (extract_memories) and real memory_store writes even for a
    run whose own report claims "nothing was actually changed". Defaults to
    False: omitting it reproduces today's exact behavior for every existing
    caller (mazu chat and mazu-web's chat have no dry-run concept at all).
    """
    if dry_run:
        memory_store.close()
        return 0

    inserted, extracted = insert_extracted_memories(
        memory_store, session_id, messages, model=model, usage_store=usage_store, belief_log=belief_log
    )

    if inserted:
        print(f"[memory] saved {inserted} new memories from this session")

    summary = extracted[0]["title"] if extracted else ""
    memory_store.end_session(session_id, task_summary=summary)
    memory_store.close()
    return inserted
