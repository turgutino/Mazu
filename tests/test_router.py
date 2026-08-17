"""Tests for the Learning Model Router (mazu/runs/router.py): classify_task's
keyword heuristic, model_stats_by_task_type's aggregation, and suggest_model's
sample-size gate + wording. No real API calls anywhere -- history is seeded
directly into real RunStore/UsageStore instances, matching this codebase's
established convention (test_runs_store.py, test_usage_store.py).
"""

from pathlib import Path

import pytest

from mazu.runs.router import (
    MIN_SAMPLES_FOR_SUGGESTION,
    ModelStats,
    classify_task,
    model_stats_by_task_type,
    suggest_model,
)
from mazu.runs.store import RunStore
from mazu.usage.store import UsageStore


@pytest.fixture
def run_store(tmp_path: Path) -> RunStore:
    s = RunStore(tmp_path / "runs.db")
    yield s
    s.close()


@pytest.fixture
def usage_store(tmp_path: Path) -> UsageStore:
    s = UsageStore(tmp_path / "usage.db")
    yield s
    s.close()


def _seed_branch(
    run_store: RunStore,
    usage_store: UsageStore,
    run_id: str,
    task: str,
    model: str,
    group_id: str,
    cost: float,
    test_passed: bool | None,
) -> None:
    run_store.start(run_id, task, model, 15, 1, True, None, None, False)
    run_store.finish(run_id, status="completed", stop_reason="end_turn")
    run_store.set_explore_outcome(run_id, group_id, test_passed)
    provider, _, model_name = model.partition(":")
    usage_store.log("run", run_id, provider, model_name or provider, 1000, 100, cost)


# ---------------------------------------------------------------------------
# classify_task
# ---------------------------------------------------------------------------


def test_classify_task_bugfix():
    assert classify_task("fix the bug where login crashes") == "bugfix"


def test_classify_task_feature():
    assert classify_task("add support for dark mode") == "feature"


def test_classify_task_refactor():
    assert classify_task("refactor the parser to simplify the code") == "refactor"


def test_classify_task_test():
    assert classify_task("add a unit test for the parser") == "test"


def test_classify_task_docs():
    assert classify_task("update the README documentation") == "docs"


def test_classify_task_general_catch_all():
    assert classify_task("look at the weather today") == "general"


def test_classify_task_known_misclassification_is_expected_v1_behavior():
    """Documented limitation, not a bug: first-match-wins over the ordered keyword
    list means a task mentioning both "fix" and "docs" lands on whichever category
    is checked first (bugfix), even though "fix a typo in the docs" arguably reads
    as a docs task to a human. Locking this in as an explicit, asserted test keeps
    the limitation honest and visible rather than silently changing behavior later.
    """
    assert classify_task("fix a typo in the docs") == "bugfix"


def test_classify_task_never_returns_none():
    assert classify_task("") == "general"
    assert classify_task("asdkjfh qwoeiruqwoe") == "general"


# ---------------------------------------------------------------------------
# model_stats_by_task_type
# ---------------------------------------------------------------------------


def test_model_stats_groups_by_model_and_computes_win_rate(run_store, usage_store):
    # Group 1: deepseek cheaper, no test command -> deepseek wins on cost.
    _seed_branch(run_store, usage_store, "r1", "fix the bug", "deepseek:deepseek-chat", "g1", 0.01, None)
    _seed_branch(run_store, usage_store, "r2", "fix the bug", "anthropic:claude-sonnet-5", "g1", 0.05, None)
    # Group 2: same pairing, same result.
    _seed_branch(run_store, usage_store, "r3", "fix another bug", "deepseek:deepseek-chat", "g2", 0.01, None)
    _seed_branch(run_store, usage_store, "r4", "fix another bug", "anthropic:claude-sonnet-5", "g2", 0.05, None)

    stats = model_stats_by_task_type(run_store, usage_store)
    by_model = {s.model: s for s in stats}

    assert by_model["deepseek:deepseek-chat"].wins == 2
    assert by_model["deepseek:deepseek-chat"].total == 2
    assert by_model["anthropic:claude-sonnet-5"].wins == 0
    assert by_model["anthropic:claude-sonnet-5"].total == 2


def test_model_stats_test_pass_beats_cost_when_test_command_given(run_store, usage_store):
    # Expensive but passing beats cheap but failing, when a test command was used.
    _seed_branch(run_store, usage_store, "r1", "fix the bug", "expensive:model", "g1", 0.50, True)
    _seed_branch(run_store, usage_store, "r2", "fix the bug", "cheap:model", "g1", 0.01, False)

    stats = model_stats_by_task_type(run_store, usage_store)
    by_model = {s.model: s for s in stats}

    assert by_model["expensive:model"].wins == 1
    assert by_model["cheap:model"].wins == 0


def test_model_stats_filters_by_task_type(run_store, usage_store):
    _seed_branch(run_store, usage_store, "r1", "fix the bug", "model-a", "g1", 0.01, None)
    _seed_branch(run_store, usage_store, "r2", "fix the bug", "model-b", "g1", 0.02, None)
    _seed_branch(run_store, usage_store, "r3", "add a new feature", "model-a", "g2", 0.01, None)
    _seed_branch(run_store, usage_store, "r4", "add a new feature", "model-b", "g2", 0.02, None)

    bugfix_stats = model_stats_by_task_type(run_store, usage_store, task_type="bugfix")
    feature_stats = model_stats_by_task_type(run_store, usage_store, task_type="feature")

    assert sum(s.total for s in bugfix_stats) == 2
    assert sum(s.total for s in feature_stats) == 2


def test_model_stats_ignores_ordinary_non_explore_runs(run_store, usage_store):
    # A plain `mazu run` -- no set_explore_outcome call, explore_group_id stays NULL.
    run_store.start("r1", "fix the bug", "model-a", 15, 1, True, None, None, False)
    run_store.finish("r1", status="completed", stop_reason="end_turn")
    usage_store.log("run", "r1", "provider", "model-a", 1000, 100, 0.05)

    stats = model_stats_by_task_type(run_store, usage_store)
    assert stats == []


def test_model_stats_pass_rate_and_win_rate_properties(run_store, usage_store):
    _seed_branch(run_store, usage_store, "r1", "fix the bug", "model-a", "g1", 0.01, True)
    _seed_branch(run_store, usage_store, "r2", "fix the bug", "model-a", "g2", 0.01, False)

    stats = model_stats_by_task_type(run_store, usage_store)
    s = stats[0]
    assert s.tested == 2
    assert s.passed == 1
    assert s.pass_rate == pytest.approx(0.5)
    assert s.win_rate == pytest.approx(1.0)  # won both (only competitor in each group)


def test_model_stats_zero_history_returns_empty_list(run_store, usage_store):
    assert model_stats_by_task_type(run_store, usage_store) == []


# ---------------------------------------------------------------------------
# suggest_model
# ---------------------------------------------------------------------------


def test_suggest_model_returns_none_below_min_samples(run_store, usage_store):
    _seed_branch(run_store, usage_store, "r1", "fix the bug", "model-a", "g1", 0.01, None)
    assert MIN_SAMPLES_FOR_SUGGESTION > 1
    assert suggest_model(run_store, usage_store, "fix another bug") is None


def test_suggest_model_returns_none_for_non_explore_history(run_store, usage_store):
    for i in range(5):
        run_store.start(f"r{i}", "fix the bug", "model-a", 15, 1, True, None, None, False)
        run_store.finish(f"r{i}", status="completed", stop_reason="end_turn")
    assert suggest_model(run_store, usage_store, "fix a bug") is None


def test_suggest_model_returns_a_string_once_enough_samples_exist(run_store, usage_store):
    for i in range(MIN_SAMPLES_FOR_SUGGESTION):
        _seed_branch(run_store, usage_store, f"a{i}", "fix bug", "deepseek:deepseek-chat", f"g{i}", 0.01, None)
        _seed_branch(run_store, usage_store, f"b{i}", "fix bug", "anthropic:claude-sonnet-5", f"g{i}", 0.05, None)

    suggestion = suggest_model(run_store, usage_store, "fix the login bug")

    assert suggestion is not None
    assert "deepseek:deepseek-chat" in suggestion
    assert "never applied automatically" in suggestion


def test_suggest_model_only_counts_matching_task_type(run_store, usage_store):
    # Plenty of "feature" history, but the new task is a "bugfix" -- must not
    # borrow feature-task samples to clear the bugfix threshold.
    for i in range(5):
        _seed_branch(run_store, usage_store, f"f{i}", "add a feature", "model-a", f"g{i}", 0.01, None)
    assert suggest_model(run_store, usage_store, "fix a crash") is None
