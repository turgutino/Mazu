from mazu.memory.store import MemoryStore, _memory_similarity

# consolidate.py's FUZZY_DUPLICATE_THRESHOLD (0.5) finds near-IDENTICAL memories
# worth merging. Contradiction candidates are a different, LOWER band: "related
# enough to plausibly be about the same thing" but clearly not duplicates -- too
# similar and it's consolidate.py's job (a real duplicate isn't a contradiction),
# too dissimilar and they're just unrelated facts. Same underlying lexical
# similarity function as consolidate.py/find_duplicate, deliberately not
# embeddings (mazu/memory/embeddings.py is OpenAI-specific and unrelated to
# whatever provider Curator's own separate key uses).
MIN_SIMILARITY = 0.12
MAX_SIMILARITY = 0.5
MAX_PAIRS = 20


def find_candidate_pairs(store: MemoryStore, max_pairs: int = MAX_PAIRS) -> list[tuple[dict, dict]]:
    """Read-only. Pairs of active memories in the same category whose lexical
    similarity falls in the "related but not a near-duplicate" band -- the actual
    contradiction-or-not judgment is left to the LLM pass, this only bounds how
    many pairs it has to look at (capped at max_pairs so a large memory store
    can't blow up the cost of one pass).
    """
    rows = [dict(r) for r in store.all_active()]
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    candidates: list[tuple[dict, dict, float]] = []
    for category_rows in by_category.values():
        for i in range(len(category_rows)):
            for j in range(i + 1, len(category_rows)):
                a, b = category_rows[i], category_rows[j]
                score = _memory_similarity(a["title"], a["body"], b["title"], b["body"])
                if MIN_SIMILARITY <= score < MAX_SIMILARITY:
                    candidates.append((a, b, score))

    candidates.sort(key=lambda c: -c[2])
    return [(a, b) for a, b, _ in candidates[:max_pairs]]
