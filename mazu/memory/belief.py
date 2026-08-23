"""Live belief-maintenance: the first, cheap half of deciding whether a newly
extracted memory relates to something already stored. See docs/memory audit for
the full design; this module implements Phase 2 (local, zero-cost gray-zone
filtering) and Phase 3 (a bounded, cheap-model classification call, SHADOW MODE
ONLY -- see belief_log.py and mazu.agent.session for how the result is recorded
and, deliberately, not yet acted on).

Nothing in this module mutates a MemoryStore. find_gray_zone_match is read-only.
classify_relationship calls an LLM but never touches the database. Applying a
classification's result to actual memory rows (supersede/edit) is intentionally
NOT implemented here yet -- see decide_action()'s docstring.
"""
from mazu.llm.client import _split_model, extract_cost_tokens, run_forced_tool
from mazu.llm.pricing import estimate_cost
from mazu.memory.extraction import extraction_model
from mazu.memory.store import FUZZY_DUPLICATE_THRESHOLD, _memory_similarity

# Same band curator/analysis/conflicts.py already uses for its own (pairwise,
# all-existing-memories) contradiction-candidate search: "related enough to
# plausibly be about the same thing" but not similar enough to already be
# caught by MemoryStore.find_duplicate's fuzzy check (>= FUZZY_DUPLICATE_THRESHOLD
# is a duplicate, not a candidate for judgment). Duplicated here rather than
# imported -- mazu.curator depends on mazu.memory, not the other way around, and
# duplicating one float constant is simpler than restructuring that dependency.
MIN_SIMILARITY = 0.12

RELATIONSHIPS = ("unrelated", "duplicate", "refinement", "replacement", "contradiction", "temporary_state")
CONFIDENCES = ("low", "medium", "high")

# Bounds the number of classification calls a single extraction batch (session-end
# or one compaction event, normally 0-5 items per extract_memories' own cap) can
# trigger, independent of that cap -- a hard ceiling so this mechanism can never
# turn one event into an unbounded number of LLM calls, even if a future caller
# feeds it a larger batch than today's extraction pipeline does.
MAX_CLASSIFICATIONS_PER_BATCH = 5

CLASSIFY_SYSTEM = (
    "You compare two memories about the same coding project and classify how the "
    "NEW one relates to the OLD one that already exists."
)

CLASSIFY_INSTRUCTIONS = """You are given an OLD memory already stored about this project, and a NEW \
memory just extracted from a recent session. Both are in the same category ({category}). Decide how \
the NEW memory relates to the OLD one:

- "unrelated": they're not actually about the same thing, despite some surface word overlap.
- "duplicate": they state the same fact, just phrased differently -- nothing new.
- "refinement": the NEW one is a more specific/complete/better-phrased statement of the SAME \
underlying fact as the OLD one, not a change in what's true.
- "replacement": the NEW one describes a genuine change -- what was true before (OLD) no longer is; \
the project has moved on. Only choose this when the NEW memory clearly describes the CURRENT state, \
not a plan, a discussion, or one option among several still being considered.
- "contradiction": the NEW and OLD memories cannot both be true right now, and it is NOT clear which \
one is actually correct/current from the text alone.
- "temporary_state": the NEW memory describes a temporary, provisional, or conditional state (e.g. \
"disabled during development", "for now", "until X is fixed") rather than a permanent change to the \
underlying fact the OLD memory describes. The OLD memory likely remains true as the general/permanent \
architecture even though the NEW memory is also worth recording.

Also rate your confidence as "low", "medium", or "high" -- how sure you are of the relationship \
classification given only the text of these two memories, with no other context about the project. \
Be conservative: prefer "contradiction" or a lower confidence over confidently guessing "replacement" \
whenever the NEW memory doesn't unambiguously describe a settled, current fact.

OLD memory:
Title: {old_title}
Body: {old_body}

NEW memory:
Title: {new_title}
Body: {new_body}

Call record_relationship with your classification."""

RECORD_RELATIONSHIP_TOOL = {
    "name": "record_relationship",
    "description": "Record how the NEW memory relates to the OLD memory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relationship": {"type": "string", "enum": list(RELATIONSHIPS)},
            "confidence": {"type": "string", "enum": list(CONFIDENCES)},
            "reasoning": {"type": "string", "description": "One sentence explaining the classification."},
        },
        "required": ["relationship", "confidence"],
    },
}


def find_gray_zone_match(store, category: str, title: str, body: str):
    """Read-only, zero-API-cost. Returns the single best-matching active memory
    in `category` whose lexical similarity to (title, body) falls strictly
    between MIN_SIMILARITY and FUZZY_DUPLICATE_THRESHOLD -- "related enough to be
    worth a judgment call" but not already handled by MemoryStore.find_duplicate's
    exact/fuzzy dedup (>= FUZZY_DUPLICATE_THRESHOLD) and not so dissimilar as to
    be a different fact entirely (< MIN_SIMILARITY). Returns None if no active
    memory in this category falls in that band -- the common case for a genuinely
    new fact, which costs nothing beyond this one linear scan.

    Mirrors curator/analysis/conflicts.py's find_candidate_pairs, but scoped to
    one new candidate against all existing memories instead of all-pairs, since
    this runs per newly extracted item, not as a periodic full-store sweep.
    """
    candidates = store.conn.execute(
        "SELECT * FROM memories WHERE category = ? AND superseded_by IS NULL AND archived = 0",
        (category,),
    ).fetchall()
    best_row, best_score = None, 0.0
    for candidate in candidates:
        score = _memory_similarity(title, body, candidate["title"], candidate["body"])
        if score > best_score:
            best_row, best_score = candidate, score
    if best_row is None:
        return None, 0.0
    if MIN_SIMILARITY <= best_score < FUZZY_DUPLICATE_THRESHOLD:
        return best_row, best_score
    return None, best_score


def classify_relationship(
    old_row, new_title: str, new_body: str, model: str | None = None
) -> tuple[dict, dict]:
    """One forced-tool-call classification, using the same cheap per-provider
    model tier extract_memories() already uses (see extraction_model()) -- no new
    model tier, no new API key dependency. Returns (result dict, usage dict);
    usage is passed through so callers can log real cost, same convention as
    extract_memories(). Raises whatever run_forced_tool raises on failure --
    callers are responsible for the try/except + fallback (see
    mazu.agent.session's shadow-mode wiring), matching extract_memories' own
    "extraction is best-effort, never crashes the session" contract.
    """
    prompt = CLASSIFY_INSTRUCTIONS.format(
        category=old_row["category"],
        old_title=old_row["title"],
        old_body=old_row["body"],
        new_title=new_title,
        new_body=new_body,
    )
    result, usage = run_forced_tool(
        messages=[{"role": "user", "content": prompt}],
        system=CLASSIFY_SYSTEM,
        tool=RECORD_RELATIONSHIP_TOOL,
        model=extraction_model(model),
    )
    return result, usage


def decide_action(relationship: str, confidence: str) -> tuple[str, bool]:
    """Pure function, no side effects: maps a (relationship, confidence) pair to
    the action live belief-maintenance WOULD take, and whether that action would
    involve superseding an existing memory. Returns (proposed_action, would_supersede).

    IMPORTANT: this function's return value is currently only ever used for
    shadow-mode LOGGING (see mazu.memory.belief_log / mazu.agent.session's
    shadow wiring). No code path in this phase calls MemoryStore.supersede(),
    .edit(), or .forget() based on this result -- that is deliberately deferred
    to a later phase, once shadow-mode evidence has been reviewed. Automatically
    resolving a "contradiction" is never proposed regardless of confidence --
    the decision table treats that as always needing a human/Curator to look,
    never something a one-shot cheap classification call gets to resolve alone.
    """
    if relationship == "unrelated":
        return "insert_new", False
    if relationship == "duplicate":
        return "skip_duplicate", False
    if relationship == "refinement":
        if confidence == "high":
            return "supersede_old_with_new", True
        return "insert_new_keep_both", False
    if relationship == "replacement":
        if confidence == "high":
            return "supersede_old_with_new", True
        return "flag_for_review", False
    if relationship == "contradiction":
        return "flag_for_review", False
    if relationship == "temporary_state":
        return "insert_new_keep_both_tag_temporary", False
    # Unknown relationship (e.g. a future classifier version, or a malformed
    # response that somehow passed schema validation) -- never guess destructively.
    return "flag_for_review", False


def run_shadow_belief_batch(
    store,
    belief_log,
    session_id: str | None,
    extracted_items: list[dict],
    model: str | None = None,
    usage_store=None,
    max_classifications: int = MAX_CLASSIFICATIONS_PER_BATCH,
) -> list[dict]:
    """SHADOW MODE ONLY. For each freshly extracted memory candidate, checks for
    a gray-zone-related existing memory (free, local) and, if found, classifies
    the relationship (one bounded, cheap-model call) and logs what action WOULD
    be taken to `belief_log`. Never touches `store` -- no insert, no supersede, no
    edit, no delete. The caller (mazu.agent.session.insert_extracted_memories)
    still performs its own existing, unchanged dedup-and-insert logic completely
    independently of this function's outcome.

    Bounded to at most `max_classifications` real LLM calls regardless of how
    many items are passed in or how many land in the gray zone -- items beyond
    that cap are still logged (so the evidence log shows they were skipped, not
    silently dropped) but with relationship/confidence left None and
    proposed_action="skipped_cap".

    If classify_relationship raises for a given item, that single item is logged
    with classifier_failed=True and proposed_action="classifier_failed" -- it
    does not raise further and does not affect any other item in the batch,
    matching extract_memories' own "never crash the session" contract.
    """
    results = []
    classifications_made = 0

    for item in extracted_items:
        category = item.get("category", "")
        title = item.get("title", "")
        body = item.get("body", "")

        match, score = find_gray_zone_match(store, category, title, body)
        if match is None:
            continue

        if classifications_made >= max_classifications:
            belief_log.log_decision(
                session_id=session_id, category=category, new_title=title, new_body=body,
                related_memory_id=match["id"], related_title=match["title"], similarity_score=score,
                relationship=None, confidence=None, proposed_action="skipped_cap",
                would_supersede=False, classifier_failed=False,
                reasoning="max classification calls for this batch already reached",
            )
            continue

        classifications_made += 1
        relationship, confidence, reasoning, classifier_failed = None, None, None, False
        try:
            classification, usage = classify_relationship(match, title, body, model=model)
            relationship = classification.get("relationship")
            confidence = classification.get("confidence")
            reasoning = classification.get("reasoning")
            if usage_store is not None and usage:
                provider_name, model_name = _split_model(extraction_model(model))
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                cost_input, _, cache_read, cache_write = extract_cost_tokens(usage)
                cost = estimate_cost(
                    f"{provider_name}:{model_name}", cost_input, output_tokens,
                    cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                )
                usage_store.log(
                    "memory_belief_shadow", session_id, provider_name, model_name,
                    input_tokens, output_tokens, cost,
                    cache_read_tokens=cache_read or None, cache_write_tokens=cache_write or None,
                )
        except Exception as e:
            classifier_failed = True
            reasoning = f"classifier error: {e}"

        if classifier_failed or relationship not in RELATIONSHIPS or confidence not in CONFIDENCES:
            proposed_action, would_supersede = "classifier_failed", False
            classifier_failed = True
        else:
            proposed_action, would_supersede = decide_action(relationship, confidence)

        belief_log.log_decision(
            session_id=session_id, category=category, new_title=title, new_body=body,
            related_memory_id=match["id"], related_title=match["title"], similarity_score=score,
            relationship=relationship, confidence=confidence, proposed_action=proposed_action,
            would_supersede=would_supersede, classifier_failed=classifier_failed, reasoning=reasoning,
        )
        results.append({
            "category": category, "title": title, "related_memory_id": match["id"],
            "relationship": relationship, "confidence": confidence,
            "proposed_action": proposed_action, "would_supersede": would_supersede,
            "classifier_failed": classifier_failed,
        })

    return results
