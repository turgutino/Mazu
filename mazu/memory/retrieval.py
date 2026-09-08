from mazu.memory.bm25 import BM25
from mazu.memory.embeddings import cosine_similarity, deserialize_embedding, embed_text, embeddings_available
from mazu.memory.store import MemoryStore

# Rough proxy for a ~2000 token budget, so the injected block never dominates the context window.
CONTEXT_CHAR_BUDGET = 8000

# Equal weight between keyword overlap and semantic closeness -- a deliberately
# simple 50/50 split, not tuned against real usage data. Revisit if it turns out
# one signal should dominate.
SEMANTIC_BLEND_WEIGHT = 0.5

CATEGORY_HEADINGS = {
    "decision": "Decisions",
    "convention": "Conventions",
    "mistake": "Mistakes to avoid",
    "task_outcome": "Past task outcomes",
    "fact": "Facts",
    "user_preference": "About You",
}


def _score_pool(pool: list, query: str) -> list[tuple]:
    """Scores each row in `pool` against `query` with local BM25 (zero API cost),
    and, only if semantic search is opted into (MAZU_SEMANTIC_MEMORY) and both the
    query and a given memory have a stored embedding, blended with cosine
    similarity. This is what lets a memory phrased very differently from the
    current task ("the project's database is Postgres" vs. "what does this project
    use for storage") still surface -- BM25 alone requires shared vocabulary.
    Returns a list of (row, bm25_normalized, semantic_or_None, combined) in the
    same order as `pool`. Shared by both the actual selection logic
    (`_rank_by_relevance`) and the inspection-only `explain_retrieval`.
    """
    if not query.strip() or not pool:
        return [(row, 0.0, None, 0.0) for row in pool]

    documents = [f"{row['title']} {row['body']} {row['tags'] or ''}" for row in pool]
    bm25_scores = BM25(documents).score(query)
    bm25_max = max(bm25_scores) if bm25_scores else 0.0
    bm25_normalized = [s / bm25_max if bm25_max > 0 else 0.0 for s in bm25_scores]

    semantic_scores = None
    if embeddings_available():
        query_embedding = embed_text(query)
        if query_embedding is not None:
            semantic_scores = []
            for row in pool:
                row_embedding = deserialize_embedding(row["embedding"])
                if row_embedding is not None:
                    # Real text embeddings' cosine similarity is effectively always
                    # in [0, 1] in practice; clamp any stray negative to 0 rather
                    # than letting it pull a combined score below a pure-keyword
                    # match's floor.
                    semantic_scores.append(max(0.0, cosine_similarity(query_embedding, row_embedding)))
                else:
                    semantic_scores.append(0.0)

    if semantic_scores is not None:
        combined = [
            (1 - SEMANTIC_BLEND_WEIGHT) * b + SEMANTIC_BLEND_WEIGHT * s
            for b, s in zip(bm25_normalized, semantic_scores)
        ]
        return list(zip(pool, bm25_normalized, semantic_scores, combined))

    return list(zip(pool, bm25_normalized, [None] * len(pool), bm25_normalized))


def _rank_by_relevance(pool: list, query: str, limit: int) -> list:
    """Selects and orders the top `limit` candidates from `pool` for actual
    injection into the context block. Falls back to recency order (pool is already
    recency-sorted) when there's no query or no term overlap at all.
    """
    if not query.strip() or not pool:
        return pool[:limit]

    scored = _score_pool(pool, query)
    ranked = sorted(scored, key=lambda t: t[3], reverse=True)
    top = [row for row, _, _, combined in ranked if combined > 0][:limit]
    return top if top else pool[:limit]


_INTRO_LINES = [
    "## Project Memory (auto-loaded)",
    "The following is prior context about this project from earlier sessions, "
    "ranked by relevance to the current task. Treat it as ground truth unless the "
    "current task explicitly contradicts it.",
    "",
]


def _order_floor_and_ranked(floor_rows: list, ranked: list) -> list:
    """Dedupes `floor_rows` + `ranked` into one list, floor rows first -- shared by
    build_context_block and explain_retrieval so both group/traverse rows in
    EXACTLY the same order (a floor row always precedes any ranked row of the
    same category, which is what lets the budget logic below skip a budget
    check for floor rows without ever having to look ahead).
    """
    seen_ids: set[int] = set()
    ordered = []
    for row in [*floor_rows, *ranked]:
        if row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        ordered.append(row)
    return ordered


def _render_within_budget(ordered: list, floor_ids: set) -> tuple[list[str], list[int]]:
    """The single source of truth for which rows actually make it into a real
    context block, and in what grouping/order -- both build_context_block (which
    turns this into the literal injected text) and explain_retrieval (which turns
    it into a truthful "why"/"included" answer) call this same function, so the
    two can never disagree about what a real injection would do.

    Floor rows (pinned + recent mistakes) are a guaranteed inclusion -- never
    subject to the CONTEXT_CHAR_BUDGET cutoff below -- because that's the whole
    point of pinning something. Real bug found live: before this, ALL rows
    (floor included) shared one running char_count across every category in
    whatever order categories happened to be encountered, so an earlier
    category's ranked entries could exhaust the budget before a LATER
    category's pinned row was ever reached, silently dropping it with no error
    or truncation notice anywhere -- and `mazu memory why` claimed it WAS
    included regardless, since it never modeled the budget at all. Within one
    category, floor rows always precede ranked rows in `ordered` (see
    _order_floor_and_ranked), so breaking out of a ranked row here never skips
    a later floor row in the same category.

    Returns (lines, rendered_ids) -- `lines` has no leading intro block (callers
    prepend/own that themselves); `rendered_ids` is every row that actually made
    it in, floor and ranked alike, in render order.
    """
    by_category: dict[str, list] = {}
    for row in ordered:
        by_category.setdefault(row["category"], []).append(row)

    lines: list[str] = []
    char_count = sum(len(line) for line in _INTRO_LINES)
    rendered_ids: list[int] = []

    for category, rows in by_category.items():
        header = f"### {CATEGORY_HEADINGS.get(category, category.title())}"
        header_added = False
        for row in rows:
            entry = f"- [id {row['id']}] {row['title']}: {row['body']} (tags: {row['tags'] or '-'})"
            if row["id"] not in floor_ids and char_count + len(entry) > CONTEXT_CHAR_BUDGET:
                break
            if not header_added:
                lines.append(header)
                char_count += len(header)
                header_added = True
            lines.append(entry)
            char_count += len(entry)
            rendered_ids.append(row["id"])
        if header_added:
            lines.append("")

    return lines, rendered_ids


def build_context_block(store: MemoryStore, query: str = "", limit: int = 15) -> str:
    """Pinned memories and recent mistakes are always included (a floor that never
    depends on query relevance). The remaining slots are filled by BM25-ranking the
    rest of the active memories against `query` — this is the "semantically close"
    retrieval described in the design, done entirely locally.
    """
    floor_rows = [*store.pinned(), *store.recent_by_category("mistake", limit=3)]
    floor_ids = {row["id"] for row in floor_rows}

    # No arbitrary cap here: all_active()'s own default (10,000) is effectively
    # unbounded for a single project's realistic memory volume, unlike the old
    # limit=500 which silently made anything older than the 500 most recent
    # memories invisible to ranking once the DB grew past that size.
    pool = [row for row in store.all_active() if row["id"] not in floor_ids]
    ranked = _rank_by_relevance(pool, query, limit)

    ordered = _order_floor_and_ranked(floor_rows, ranked)
    if not ordered:
        return ""

    body_lines, rendered_ids = _render_within_budget(ordered, floor_ids)
    if not rendered_ids:
        return ""

    # Only memories that actually made it into the rendered block count as
    # "retrieved" -- a row cut for char budget was considered but never surfaced.
    store.mark_retrieved(rendered_ids)

    return "\n".join([*_INTRO_LINES, *body_lines]).rstrip() + "\n"


def build_global_context_block(global_store: MemoryStore) -> str:
    """Renders every active `user_preference` entry from the global store — facts about
    the person, not the project, so no BM25 ranking or floor logic is needed (expected
    volume is small: name, language, experience level, a handful of work-style prefs).
    """
    rows = global_store.all_active()
    if not rows:
        return ""

    lines = [
        "## About You (auto-loaded, applies to every project)",
        "The following is durable context about the person you're working with, "
        "carried over from earlier sessions on other projects too.",
        "",
    ]
    char_count = sum(len(line) for line in lines)
    rendered_ids: list[int] = []
    for row in rows:
        entry = f"- [id {row['id']}] {row['title']}: {row['body']}"
        if char_count + len(entry) > CONTEXT_CHAR_BUDGET:
            break
        lines.append(entry)
        char_count += len(entry)
        rendered_ids.append(row["id"])

    global_store.mark_retrieved(rendered_ids)

    return "\n".join(lines).rstrip() + "\n"


def explain_retrieval(store: MemoryStore, query: str = "", limit: int = 15) -> list[dict]:
    """Inspection-only mirror of build_context_block's selection logic: returns per-memory
    scoring and inclusion decisions instead of a formatted block, and never mutates
    retrieval stats (no mark_retrieved call) since this isn't a real retrieval into a
    live session, just a "what would happen and why" view for `mazu memory why`.

    Real bug found live: this used to mark every floor row (pinned + recent
    mistakes) "included": True unconditionally, and every ranked row "included"
    based only on the top-`limit` cutoff -- neither ever modeled
    CONTEXT_CHAR_BUDGET at all, so `mazu memory why` could confidently say a
    pinned memory (or a ranked one within the item-count limit) WOULD be
    injected when build_context_block would actually have cut it for space.
    Now calls the exact same _render_within_budget() build_context_block itself
    uses, so the two can never disagree.
    """
    floor_rows = [*store.pinned(), *store.recent_by_category("mistake", limit=3)]
    floor_ids = {row["id"] for row in floor_rows}
    pinned_ids = {row["id"] for row in store.pinned()}

    pool = [row for row in store.all_active() if row["id"] not in floor_ids]
    scored = _score_pool(pool, query)
    ranked = sorted(scored, key=lambda t: t[3], reverse=True)

    included_from_ranked = [row for row, _, _, combined in ranked if combined > 0][:limit]
    if not included_from_ranked and pool:
        included_from_ranked = pool[:limit]

    ordered = _order_floor_and_ranked(floor_rows, included_from_ranked)
    _, rendered_ids = _render_within_budget(ordered, floor_ids)
    rendered_ids_set = set(rendered_ids)
    candidate_ranked_ids = {row["id"] for row in included_from_ranked}

    explanations = []
    for row in floor_rows:
        reason = "pinned" if row["id"] in pinned_ids else "recent mistake"
        explanations.append(
            {"row": row, "bm25": None, "semantic": None, "combined": None,
             "included": row["id"] in rendered_ids_set, "reason": reason}
        )
    for row, bm25, semantic, combined in ranked:
        is_candidate = row["id"] in candidate_ranked_ids
        included = row["id"] in rendered_ids_set
        if is_candidate and not included:
            reason = "cut for char budget"
        elif is_candidate:
            reason = "ranked"
        else:
            reason = "not included"
        explanations.append(
            {
                "row": row,
                "bm25": bm25,
                "semantic": semantic,
                "combined": combined,
                "included": included,
                "reason": reason,
            }
        )
    return explanations
