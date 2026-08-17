"""Learning Model Router: the historical analog of mazu.agent.explore's
format_explore_report. Instead of ranking one in-memory list of branch results from
a single `mazu explore` call, this aggregates every explore-tagged row ever recorded
in RunStore (across many past `mazu explore` invocations, this project only) grouped
by model, to answer "which model has tended to win explore comparisons here,
optionally for a given kind of task."

Project-scoped only (reads the same .mazu/runs.db RunStore already uses) -- no
cross-project learning, matching RunStore's own project-scoped design (unlike
UsageStore, which is intentionally global at ~/.mazu/usage.db). "Which model wins
bug-fix tasks" in one codebase/language is not a valid inference for a different one.

NEVER used to select a model automatically. suggest_model() returns a printable
string or None -- every caller only ever prints it. Nothing in this module is
imported by mazu.llm.client.default_model() or ensure_api_key(), and nothing here
ever reassigns a `model` variable on a caller's behalf.

Task classification (classify_task) is a crude, English-only, first-match-wins
keyword heuristic, not real intent understanding -- an LLM-based classifier was
deliberately rejected for v1 since it would add a real API call to every plain
`mazu run`/`mazu chat` invocation, against this project's established minimum-cost
ethos. A task mentioning multiple keywords ("fix the docs typo") is classified by
whichever category is checked first, which will sometimes be the "wrong" one --
this is a known, accepted limitation, not a hidden one. Ranking policy changes (in
mazu.agent.explore._rank_key) are NOT retroactively applied to facts that weren't
recorded at the time -- only the stored facts (test_passed) get re-weighted by a
new rule; a new signal that wasn't captured historically can't be reconstructed.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from mazu.agent.explore import _rank_key
from mazu.runs.store import RunStore
from mazu.usage.store import UsageStore

MIN_SAMPLES_FOR_SUGGESTION = 3  # don't suggest from 1-2 data points

# Checked in order, first match wins. "test"/"docs" come before "feature" so "add
# tests for X" or "add docs for X" don't fall into "feature" via the word "add".
_TASK_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("bugfix", ("fix", "bug", "error", "crash", "broken", "regression", "fails", "failing")),
    ("test", ("test", "pytest", "unit test", "coverage")),
    ("docs", ("docs", "documentation", "readme", "docstring", "comment")),
    ("refactor", ("refactor", "clean up", "cleanup", "simplify", "rename", "reorganize", "restructure")),
    ("feature", ("add", "implement", "new", "support", "create", "build")),
]

TASK_TYPES = [t for t, _ in _TASK_TYPE_KEYWORDS] + ["general"]


def classify_task(task: str) -> str:
    """Never returns None -- 'general' is the catch-all so every GROUP BY task_type
    query stays total, not partial. See module docstring for known limitations.
    """
    lowered = task.lower()
    for task_type, keywords in _TASK_TYPE_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return task_type
    return "general"


@dataclass
class ModelStats:
    model: str
    total: int = 0
    wins: int = 0
    passed: int = 0   # count where test_passed == 1
    tested: int = 0   # count where test_passed is not NULL (1 or 0)
    total_cost: float = 0.0
    avg_cost: float = field(default=0.0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.tested if self.tested else None


def model_stats_by_task_type(
    run_store: RunStore, usage_store: UsageStore, task_type: str | None = None
) -> list[ModelStats]:
    """Groups every explore-tagged run (explore_group_id IS NOT NULL) by model. If
    task_type is given, filters to rows whose classify_task(task) matches it.
    "Wins" are recomputed live per explore_group_id using the same _rank_key
    mazu.agent.explore.format_explore_report uses -- never a stored value (see
    module docstring on why: a ranking policy should stay recomputable, not frozen
    into history the moment it's written).
    """
    rows = run_store.conn.execute(
        "SELECT id, task, model, explore_group_id, test_passed, status "
        "FROM runs WHERE explore_group_id IS NOT NULL AND model IS NOT NULL"
    ).fetchall()

    if task_type is not None:
        rows = [r for r in rows if classify_task(r["task"]) == task_type]

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r["explore_group_id"]].append(r)

    per_model: dict[str, ModelStats] = {}
    cost_cache: dict[str, float] = {}

    def _cost(run_id: str) -> float:
        if run_id not in cost_cache:
            cost_cache[run_id] = usage_store.summary(session_id=run_id)["total_cost"] or 0.0
        return cost_cache[run_id]

    for group_rows in groups.values():
        has_test_command = any(r["test_passed"] is not None for r in group_rows)
        ranked = sorted(
            group_rows,
            key=lambda r: _rank_key(
                _cost(r["id"]),
                None if r["test_passed"] is None else bool(r["test_passed"]),
                has_test_command,
            ),
        )
        winner_id = ranked[0]["id"] if ranked else None
        for r in group_rows:
            stats = per_model.setdefault(r["model"], ModelStats(model=r["model"]))
            stats.total += 1
            stats.total_cost += _cost(r["id"])
            if r["id"] == winner_id:
                stats.wins += 1
            if r["test_passed"] is not None:
                stats.tested += 1
                if r["test_passed"]:
                    stats.passed += 1

    for stats in per_model.values():
        stats.avg_cost = stats.total_cost / stats.total if stats.total else 0.0

    return sorted(per_model.values(), key=lambda s: (-s.win_rate, s.avg_cost))


def suggest_model(run_store: RunStore, usage_store: UsageStore, task: str) -> str | None:
    """Returns a printable suggestion string, or None if there isn't enough history
    (fewer than MIN_SAMPLES_FOR_SUGGESTION matching explore-branch outcomes for this
    task type) to say anything meaningful. Callers must only ever print this --
    never assign it to a `model` variable that gets passed to run_autonomous.
    """
    task_type = classify_task(task)
    stats = model_stats_by_task_type(run_store, usage_store, task_type=task_type)
    total_samples = sum(s.total for s in stats)
    if total_samples < MIN_SAMPLES_FOR_SUGGESTION or not stats:
        return None
    top = stats[0]
    return (
        f"[router] Based on {total_samples} past '{task_type}' exploration(s) in this "
        f"project, {top.model} won {top.wins}/{top.total} ({top.win_rate:.0%}) at "
        f"~${top.avg_cost:.4f} avg. Consider: --model {top.model} "
        "(this is only a suggestion -- it is never applied automatically)."
    )
