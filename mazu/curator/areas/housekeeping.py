from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.curator.analysis.branches import find_abandoned_branches
from mazu.curator.areas import Area
from mazu.curator.prompts import HOUSEKEEPING_TASK_PROMPT

_MIN_ACTIONS_FOR_SIGNAL = 10


def compute_signal(root: Path) -> int:
    """Zero-API-cost signal: any abandoned branch found, OR enough new action-log
    volume to be worth a pattern-mining pass. Deliberately coarse (this area is
    read-mostly/advisory, not high-stakes) -- a false-negative here just means one
    project waits an extra cycle before Curator looks again, not a correctness
    problem."""
    abandoned = find_abandoned_branches(root)
    if abandoned:
        return len(abandoned)

    action_log_path = root / ".mazu" / "action_log.db"
    if not action_log_path.exists():
        return 0
    action_log = ActionLogStore(action_log_path)
    try:
        total_actions = sum(s["action_count"] for s in action_log.list_sessions(limit=200))
    finally:
        action_log.close()
    return 1 if total_actions >= _MIN_ACTIONS_FOR_SIGNAL else 0


HOUSEKEEPING_AREA = Area(
    name="housekeeping", min_interval_days=14, signal_fn=compute_signal, task_prompt=HOUSEKEEPING_TASK_PROMPT
)
