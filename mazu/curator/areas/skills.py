from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.curator.areas import Area
from mazu.curator.prompts import SKILLS_TASK_PROMPT
from mazu.skills.manager import SkillManager

_MIN_RUNS_FOR_SIGNAL = 3
_FAILURE_RATE_THRESHOLD = 0.5


def _skills_db_paths(root: Path) -> tuple[Path, Path]:
    return root / ".mazu" / "skills", root / ".mazu" / "action_log.db"


def compute_signal(root: Path) -> int:
    """Zero-API-cost signal: any skill with a rough failure-rate over the sample
    threshold, OR any new skill saved since the process started (mtime-based --
    approximated here by simply counting skills that exist, since a brand-new
    project has none and this function is only ever consulted after
    curator_configured() is true). Kept intentionally simple for Phase 2 --
    real duplicate-description detection happens inside the LLM pass itself,
    not in this cheap pre-check."""
    skills_dir, action_log_path = _skills_db_paths(root)
    if not skills_dir.exists():
        return 0
    manager = SkillManager(root)
    metas = manager.list()
    if not metas:
        return 0
    if not action_log_path.exists():
        return 0
    action_log = ActionLogStore(action_log_path)
    try:
        outcomes = action_log.skill_run_outcomes()
    finally:
        action_log.close()
    signal = 0
    for counts in outcomes.values():
        total = counts["ok"] + counts["error"]
        if total >= _MIN_RUNS_FOR_SIGNAL and counts["error"] / total >= _FAILURE_RATE_THRESHOLD:
            signal += 1
    return signal


SKILLS_AREA = Area(
    name="skills", min_interval_days=1, signal_fn=compute_signal, task_prompt=SKILLS_TASK_PROMPT
)
