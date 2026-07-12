from mazu.memory.store import FUZZY_DUPLICATE_THRESHOLD, MemoryStore, _memory_similarity


def _find_clusters(rows: list, threshold: float) -> list[list]:
    """Groups `rows` into clusters of near-duplicates using pairwise Jaccard
    similarity (via _memory_similarity) as edges and connected components as
    clusters. Transitive: if A~B and B~C both clear the threshold, all three land
    in one cluster even if A and C aren't directly similar enough on their own --
    this is deliberate (A and C are still the same underlying fact, just related
    through B), not a bug. Rows with no similar neighbor are excluded from the
    result entirely (a cluster of size 1 isn't a duplicate of anything).
    """
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            score = _memory_similarity(
                rows[i]["title"], rows[i]["body"], rows[j]["title"], rows[j]["body"]
            )
            if score >= threshold:
                union(i, j)

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(rows[i])

    return [g for g in groups.values() if len(g) > 1]


def find_duplicate_clusters(
    store: MemoryStore, threshold: float = FUZZY_DUPLICATE_THRESHOLD
) -> list[list]:
    """Read-only: finds groups of 2+ active memories that look like the same fact
    restated. Clustering runs separately per category -- a 'decision' and a
    'mistake' with similar wording are not duplicates of each other, they're two
    different kinds of facts that happen to share vocabulary. Never modifies the
    store; safe to call for a dry-run preview.
    """
    memories = store.all_active()
    by_category: dict[str, list] = {}
    for m in memories:
        by_category.setdefault(m["category"], []).append(m)

    clusters = []
    for rows in by_category.values():
        clusters.extend(_find_clusters(rows, threshold))
    return clusters


def apply_consolidation(store: MemoryStore, clusters: list[list]) -> list[dict]:
    """For each cluster, keeps the most-recently-created memory (by created_at) as
    the survivor and marks every other member as superseded by it, via the
    existing supersede() mechanism -- nothing is deleted, the audit trail is
    preserved, and all_active()/context injection immediately stop showing the
    superseded rows. Returns one summary dict per cluster consolidated.
    """
    summary = []
    for cluster in clusters:
        cluster_sorted = sorted(cluster, key=lambda r: r["created_at"])
        survivor = cluster_sorted[-1]
        losers = cluster_sorted[:-1]
        for loser in losers:
            store.supersede(loser["id"], survivor["id"])
        summary.append(
            {
                "category": survivor["category"],
                "survivor_id": survivor["id"],
                "survivor_title": survivor["title"],
                "superseded": [{"id": r["id"], "title": r["title"]} for r in losers],
            }
        )
    return summary
