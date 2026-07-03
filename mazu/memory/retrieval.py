from mazu.memory.bm25 import BM25
from mazu.memory.store import MemoryStore

# Rough proxy for a ~2000 token budget, so the injected block never dominates the context window.
CONTEXT_CHAR_BUDGET = 8000

CATEGORY_HEADINGS = {
    "decision": "Decisions",
    "convention": "Conventions",
    "mistake": "Mistakes to avoid",
    "task_outcome": "Past task outcomes",
    "fact": "Facts",
    "user_preference": "About You",
}


def _rank_by_relevance(pool: list, query: str, limit: int) -> list:
    """Rank the candidate pool against a query using local BM25 (zero API cost).
    Falls back to recency order (pool is already recency-sorted) when there's no
    query or no term overlap at all.
    """
    if not query.strip() or not pool:
        return pool[:limit]

    documents = [f"{row['title']} {row['body']} {row['tags'] or ''}" for row in pool]
    scores = BM25(documents).score(query)
    ranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)
    top = [row for row, score in ranked if score > 0][:limit]
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
        lines.append("")

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
    for row in rows:
        entry = f"- [id {row['id']}] {row['title']}: {row['body']}"
        if char_count + len(entry) > CONTEXT_CHAR_BUDGET:
            break
        lines.append(entry)
        char_count += len(entry)

    return "\n".join(lines).rstrip() + "\n"
