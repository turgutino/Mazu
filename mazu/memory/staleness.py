from datetime import datetime, timedelta, timezone

from mazu.memory.store import MemoryStore

DEFAULT_STALE_DAYS = 30


def find_stale_candidates(store: MemoryStore, min_days: int = DEFAULT_STALE_DAYS) -> list:
    """Read-only: finds active memories that haven't been retrieved into a real
    session's context in at least `min_days` days -- candidates worth a human (or
    `mazu memory stale --auto`) reviewing, never a definitive "this is wrong" list.
    A stale last_used_at means "hasn't come up recently", which can mean the fact
    genuinely went out of date, or just that it's a real, still-true fact nobody's
    asked about lately -- this function can't tell the difference, only surface
    candidates.

    Two categories of active memory are deliberately excluded, because
    build_context_block() (mazu/memory/retrieval.py) always injects them
    regardless of query relevance, so their last_used_at is never a meaningful
    staleness signal -- it gets refreshed every single session no matter what:
      - pinned memories (the human already decided these are always relevant)
      - the *N most recent* memories in the "mistake" category (build_context_block's
        own floor, currently N=3 via recent_by_category("mistake", limit=3))
    Older "mistake" rows beyond that top-3 floor DO fall through to normal
    relevance-ranked retrieval, so they're still eligible for staleness here.

    A memory younger than `min_days` is never flagged even with a NULL
    last_used_at -- it simply hasn't had a fair chance to be retrieved yet, that's
    not staleness.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_days)).isoformat()

    floor_mistake_ids = {row["id"] for row in store.recent_by_category("mistake", limit=3)}

    candidates = []
    for row in store.all_active():
        if row["pinned"]:
            continue
        if row["id"] in floor_mistake_ids:
            continue
        if row["created_at"] >= cutoff:
            continue
        if row["last_used_at"] is not None and row["last_used_at"] >= cutoff:
            continue
        candidates.append(row)
    return candidates


def apply_archival(store: MemoryStore, candidates: list) -> list[dict]:
    """Archives every candidate (store.archive() -- reversible, not a delete).
    Returns one summary dict per row actually archived, for a caller to report
    back what changed.
    """
    summary = []
    for row in candidates:
        if store.archive(row["id"]):
            summary.append({"id": row["id"], "category": row["category"], "title": row["title"]})
    return summary
