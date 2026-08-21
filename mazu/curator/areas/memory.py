from pathlib import Path

from mazu.curator.areas import Area
from mazu.curator.prompts import MEMORY_TASK_PROMPT
from mazu.memory.consolidate import find_duplicate_clusters
from mazu.memory.staleness import find_stale_candidates
from mazu.memory.store import MemoryStore


def _memory_db_path(root: Path) -> Path:
    return root / ".mazu" / "memory.db"


def compute_signal(root: Path) -> int:
    """A local, zero-API-cost count of "is there anything worth Curator's attention
    here" -- stale candidates + duplicate clusters. A brand-new/empty project has
    no memory.db yet; MemoryStore's own __init__ creates it, which is harmless
    (just an empty sqlite file) but this function still returns 0 for it either way."""
    db_path = _memory_db_path(root)
    store = MemoryStore(db_path)
    try:
        signal = len(find_stale_candidates(store)) + len(find_duplicate_clusters(store))
    finally:
        store.close()
    return signal


MEMORY_AREA = Area(
    name="memory", min_interval_days=1, signal_fn=compute_signal, task_prompt=MEMORY_TASK_PROMPT
)
