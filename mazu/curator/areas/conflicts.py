from pathlib import Path

from mazu.curator.analysis.conflicts import find_candidate_pairs
from mazu.curator.areas import Area
from mazu.curator.prompts import CONFLICTS_TASK_PROMPT
from mazu.memory.store import MemoryStore


def _memory_db_path(root: Path) -> Path:
    return root / ".mazu" / "memory.db"


def compute_signal(root: Path) -> int:
    db_path = _memory_db_path(root)
    store = MemoryStore(db_path)
    try:
        signal = len(find_candidate_pairs(store))
    finally:
        store.close()
    return signal


CONFLICTS_AREA = Area(
    name="conflicts", min_interval_days=7, signal_fn=compute_signal, task_prompt=CONFLICTS_TASK_PROMPT
)
