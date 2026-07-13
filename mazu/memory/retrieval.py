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

    seen_ids: set[int] = set()
    ordered = []
    for row in [*floor_rows, *ranked]:
        if row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        ordered.append(row)

    if not ordered:
        return ""

    by_category: dict[str, list] = {}
    for row in ordered:
        by_category.setdefault(row["category"], []).append(row)

    lines = [
        "## Project Memory (auto-loaded)",
        "The following is prior context about this project from earlier sessions, "
        "ranked by relevance to the current task. Treat it as ground truth unless the "
        "current task explicitly contradicts it.",
        "",
    ]
    char_count = sum(len(line) for line in lines)
    rendered_ids: list[int] = []

    for category, rows in by_category.items():
        header = f"### {CATEGORY_HEADINGS.get(category, category.title())}"
        lines.append(header)
        char_count += len(header)
        for row in rows:
            entry = f"- [id {row['id']}] {row['title']}: {row['body']} (tags: {row['tags'] or '-'})"
            if char_count + len(entry) > CONTEXT_CHAR_BUDGET:
                break
            lines.append(entry)
            char_count += len(entry)
            rendered_ids.append(row["id"])
        lines.append("")

    # Only memories that actually made it into the rendered block count as
    # "retrieved" -- a row cut for char budget was considered but never surfaced.
    store.mark_retrieved(rendered_ids)

    return "\n".join(lines).rstrip() + "\n"


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
    included_ranked_ids = {row["id"] for row in included_from_ranked}

    explanations = []
    for row in floor_rows:
        reason = "pinned" if row["id"] in pinned_ids else "recent mistake"
        explanations.append(
            {"row": row, "bm25": None, "semantic": None, "combined": None,
             "included": True, "reason": reason}
        )
    for row, bm25, semantic, combined in ranked:
        explanations.append(
            {
                "row": row,
                "bm25": bm25,
                "semantic": semantic,
                "combined": combined,
                "included": row["id"] in included_ranked_ids,
                "reason": "ranked" if row["id"] in included_ranked_ids else "not included",
            }
        )
    return explanations
