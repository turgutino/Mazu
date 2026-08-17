"""Tests for `mazu explore` (mazu/agent/explore.py): forking N branches from one
checkpoint into real git worktrees, running the same task on each with a different
model in parallel, and reporting a ranked comparison. `run_autonomous` itself is
monkeypatched to a fast fake (no real API calls, matching test_council.py's
convention) that still exercises the real worktree creation, real RunStore/
UsageStore rows, and the real shared cost tracker.
"""

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import mazu.agent.explore as explore_module
from mazu.agent.explore import (
    MAX_EXPLORE_APPROACHES,
    _slugify_model,
    format_explore_report,
    run_explore,
)
from mazu.checkpoint.manager import CheckpointManager
from mazu.cli import main
from mazu.runs.store import RunStore
from mazu.usage.store import UsageStore


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def hello():\n    pass\n")
    return tmp_path


@pytest.fixture
def checkpoint_manager(project: Path) -> CheckpointManager:
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")
    return manager


# ---------------------------------------------------------------------------
# _slugify_model
# ---------------------------------------------------------------------------


def test_slugify_model_basic():
    used: set[str] = set()
    assert _slugify_model("anthropic:claude-sonnet-5", used) == "anthropic-claude-sonnet-5"


def test_slugify_model_collision_gets_a_suffix():
    used: set[str] = set()
    first = _slugify_model("deepseek:deepseek-chat", used)
    second = _slugify_model("deepseek:deepseek-chat", used)
    assert first == "deepseek-deepseek-chat"
    assert second == "deepseek-deepseek-chat-2"
    assert first != second


# ---------------------------------------------------------------------------
# format_explore_report
# ---------------------------------------------------------------------------


def _fake_result(model, branch, cost, status="completed", test_passed=None, error=None):
    return {
        "model": model,
        "branch_name": branch,
        "worktree_path": Path(f"/fake/{branch}"),
        "run_row": {"status": status, "stop_reason": "end_turn", "last_step": 3, "max_steps": 15, "checkpoints_created": 2},
        "cost": cost,
        "test_passed": test_passed,
        "test_output_tail": "",
        "error": error,
    }


def test_format_report_ranks_by_cost_when_no_test_command():
    results = [_fake_result("gpt", "b-gpt", 0.05), _fake_result("claude", "b-claude", 0.01)]
    report = format_explore_report(results, test_command=None)
    assert report.index("b-claude") < report.index("b-gpt")


def test_format_report_ranks_passing_tests_first():
    results = [
        _fake_result("gpt", "b-gpt", 0.01, test_passed=False),
        _fake_result("claude", "b-claude", 0.05, test_passed=True),
    ]
    report = format_explore_report(results, test_command="pytest")
    # claude cost more but passed -- must still rank above the cheaper failing branch.
    assert report.index("b-claude") < report.index("b-gpt")


def test_format_report_shows_error_branches_without_crashing():
    results = [_fake_result("gpt", "b-gpt", 0.0, error="boom")]
    report = format_explore_report(results, test_command=None)
    assert "ERROR: boom" in report


def test_format_report_includes_adopt_command():
    results = [_fake_result("gpt", "b-gpt", 0.01)]
    report = format_explore_report(results, test_command=None)
    # `cd <worktree>` is the primary, always-usable adoption path (a `git checkout
    # <branch>` in the origin repo fails while the worktree still exists) -- the
    # worktree-removal + checkout alternative is offered as a secondary option.
    assert "cd " in report
    assert "git worktree remove" in report
    assert "git checkout b-gpt" in report


# ---------------------------------------------------------------------------
# run_explore -- real worktrees, faked run_autonomous
# ---------------------------------------------------------------------------


def _make_fake_run_autonomous(cost_per_branch: dict):
    """Returns a fake run_autonomous that writes a real RunStore row (matching
    what a real run does) and feeds a fixed cost into the shared tracker, without
    making any real API calls -- fast, deterministic, and still exercises the
    real read-back path (result_run_store.get / usage_store.summary) explore.py
    uses after run_autonomous returns.
    """

    def _fake(
        registry,
        task,
        session_id,
        checkpoint_manager,
        memory_store=None,
        global_memory_store=None,
        skill_manager=None,
        max_steps=15,
        allow_shell=False,
        max_cost=None,
        model=None,
        usage_store=None,
        action_log_store=None,
        run_store=None,
        resume_messages=None,
        origin_checkpoint_id=None,
        branch_name=None,
        shared_cost_tracker=None,
        **kwargs,
    ):
        cost = cost_per_branch.get(model, 0.01)
        run_store.start(session_id, task, model, max_steps, 1, allow_shell, None, max_cost, False, branch_name=branch_name)
        run_store.update_progress(session_id, 4, checkpoint_id="cp_000004")
        if usage_store is not None:
            usage_store.log("run", session_id, "fake", model or "fake-model", 100, 50, cost)
        if shared_cost_tracker is not None:
            shared_cost_tracker.add_and_check(cost)
        run_store.finish(session_id, status="completed", stop_reason="end_turn")
        # Mirror real run_autonomous: it closes these itself in its own finally.
        run_store.close()
        if usage_store is not None:
            usage_store.close()
        if action_log_store is not None:
            action_log_store.close()
        if global_memory_store is not None:
            global_memory_store.close()

    return _fake


def test_run_explore_creates_one_worktree_per_model(project, checkpoint_manager, monkeypatch):
    monkeypatch.setattr(
        explore_module, "run_autonomous", _make_fake_run_autonomous({"model-a": 0.01, "model-b": 0.02})
    )

    results = run_explore(
        "do the thing", models=["model-a", "model-b"], root=project, checkpoint_manager=checkpoint_manager
    )

    assert len(results) == 2
    for r in results:
        assert r["worktree_path"].exists()
        assert r["error"] is None
    models_seen = {r["model"] for r in results}
    assert models_seen == {"model-a", "model-b"}


def test_run_explore_branches_have_distinct_names(project, checkpoint_manager, monkeypatch):
    monkeypatch.setattr(explore_module, "run_autonomous", _make_fake_run_autonomous({}))

    results = run_explore("task", models=["model-a", "model-b"], root=project, checkpoint_manager=checkpoint_manager)

    branch_names = [r["branch_name"] for r in results]
    assert len(set(branch_names)) == 2


def test_run_explore_reports_real_cost_from_usage_store(project, checkpoint_manager, monkeypatch):
    monkeypatch.setattr(
        explore_module, "run_autonomous", _make_fake_run_autonomous({"model-a": 0.1234})
    )

    results = run_explore("task", models=["model-a"], root=project, checkpoint_manager=checkpoint_manager)

    assert results[0]["cost"] == pytest.approx(0.1234)


def test_run_explore_shared_cost_tracker_is_genuinely_shared(project, checkpoint_manager, monkeypatch):
    """The whole point of shared_cost_tracker: one branch's spend must be visible
    to (and able to cut short) a sibling branch, not just constructed and ignored.
    """
    captured_trackers = []
    real_fake = _make_fake_run_autonomous({"model-a": 0.6, "model-b": 0.6})

    def _fake_with_capture(*args, shared_cost_tracker=None, **kwargs):
        captured_trackers.append(shared_cost_tracker)
        return real_fake(*args, shared_cost_tracker=shared_cost_tracker, **kwargs)

    monkeypatch.setattr(explore_module, "run_autonomous", _fake_with_capture)

    run_explore(
        "task", models=["model-a", "model-b"], root=project, checkpoint_manager=checkpoint_manager, max_cost=1.0
    )

    # Both branches were handed the SAME tracker instance -- proving the budget is
    # shared, not one-per-branch.
    assert captured_trackers[0] is captured_trackers[1]
    assert captured_trackers[0].total == pytest.approx(1.2)
    assert captured_trackers[0].is_exhausted() is True


def test_run_explore_mirrors_outcomes_into_the_origin_projects_run_store(project, checkpoint_manager, monkeypatch):
    """Real bug caught in live testing: each branch's RunStore row is written into
    ITS OWN worktree's .mazu/runs.db (necessary for that worktree's own `mazu
    runs`), but mazu/runs/router.py's suggestions and `mazu router stats` query the
    ORIGIN project's own .mazu/runs.db -- which never saw these rows before this
    fix, so `mazu router stats` silently showed "no history" forever, even right
    after real explore runs. This asserts the mirror actually lands.
    """
    monkeypatch.setattr(
        explore_module, "run_autonomous", _make_fake_run_autonomous({"model-a": 0.02, "model-b": 0.03})
    )

    results = run_explore(
        "task", models=["model-a", "model-b"], root=project, checkpoint_manager=checkpoint_manager
    )

    origin_run_store = RunStore(project / ".mazu" / "runs.db")
    for r in results:
        mirrored = origin_run_store.get(r["session_id"])
        assert mirrored is not None, f"no mirrored row in the origin project for {r['model']}"
        assert mirrored["model"] == r["model"]
        assert mirrored["branch_name"] == r["branch_name"]
        assert mirrored["explore_group_id"] is not None
        assert mirrored["status"] == "completed"
        # Real bug: start()/finish() alone leave these at their schema defaults
        # (0/None) -- the mirror must copy the branch's REAL final progress, not
        # just create/finish a hollow row.
        assert mirrored["last_step"] == 4
        assert mirrored["checkpoints_created"] == 1
        assert mirrored["last_checkpoint_id"] == "cp_000004"
    # Both branches from one run_explore() call share the same explore_group_id in
    # the ORIGIN's store too, not just each worktree's own (test_run_explore_branches_
    # have_distinct_names already covers the worktree side).
    group_ids = {origin_run_store.get(r["session_id"])["explore_group_id"] for r in results}
    assert len(group_ids) == 1
    origin_run_store.close()


def test_run_explore_works_with_no_pre_existing_checkpoint(project, monkeypatch):
    """Real bug caught in live testing: a fresh project that never ran `mazu
    checkpoint`/`mazu run` has an empty index.json -- explore must auto-create a
    baseline checkpoint to fork from rather than requiring the user to do that
    manually first. Deliberately does NOT use the `checkpoint_manager` fixture
    (which already snapshots once) so this project genuinely starts with zero.
    """
    manager = CheckpointManager(project)
    assert manager.list_checkpoints() == []
    monkeypatch.setattr(explore_module, "run_autonomous", _make_fake_run_autonomous({}))

    results = run_explore("task", models=["model-a"], root=project, checkpoint_manager=manager)

    assert len(results) == 1
    assert results[0]["error"] is None
    assert len(manager.list_checkpoints()) == 1


def test_run_explore_invalid_base_checkpoint_raises_cleanly(project, checkpoint_manager):
    with pytest.raises(ValueError):
        run_explore("task", models=["model-a"], root=project, checkpoint_manager=checkpoint_manager, from_checkpoint_id="cp_nonexistent")


def test_run_explore_one_branch_failing_does_not_sink_the_others(project, checkpoint_manager, monkeypatch):
    def _fake(registry, task, session_id, checkpoint_manager, run_store=None, model=None, branch_name=None, **kwargs):
        if model == "model-bad":
            raise RuntimeError("simulated crash")
        run_store.start(session_id, task, model, 15, 1, True, None, None, False, branch_name=branch_name)
        run_store.finish(session_id, status="completed", stop_reason="end_turn")
        run_store.close()

    monkeypatch.setattr(explore_module, "run_autonomous", _fake)

    results = run_explore(
        "task", models=["model-bad", "model-good"], root=project, checkpoint_manager=checkpoint_manager
    )

    by_model = {r["model"]: r for r in results}
    assert by_model["model-bad"]["error"] is not None
    assert by_model["model-good"]["error"] is None


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_explore_rejects_too_many_approaches(project, monkeypatch):
    monkeypatch.chdir(project)
    runner = CliRunner()
    result = runner.invoke(
        main, ["explore", "task", "--approaches", str(MAX_EXPLORE_APPROACHES + 1), "--models", "a,b,c,d,e,f"]
    )
    assert result.exit_code != 0
    assert "exceeds the maximum" in result.output


def test_cli_explore_requires_models_when_approaches_over_one(project, monkeypatch):
    monkeypatch.chdir(project)
    runner = CliRunner()
    result = runner.invoke(main, ["explore", "task", "--approaches", "2"])
    assert result.exit_code != 0
    assert "--models is required" in result.output


def test_cli_explore_rejects_model_count_mismatch(project, monkeypatch):
    monkeypatch.chdir(project)
    runner = CliRunner()
    result = runner.invoke(main, ["explore", "task", "--approaches", "3", "--models", "a,b"])
    assert result.exit_code != 0
    assert "lists 2 model(s) but --approaches is 3" in result.output


def test_cli_explore_happy_path_reaches_run_explore(project, monkeypatch):
    monkeypatch.chdir(project)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    captured = {}

    def _fake_run_explore(task, models, root, checkpoint_manager, from_checkpoint_id=None, max_cost=None, test_command=None, max_steps=15):
        captured.update(
            task=task, models=models, from_checkpoint_id=from_checkpoint_id, max_cost=max_cost, test_command=test_command
        )
        return []

    monkeypatch.setattr("mazu.cli.run_explore", _fake_run_explore)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "explore", "fix the bug",
            "--approaches", "2",
            "--models", "anthropic:claude-sonnet-5,deepseek:deepseek-chat",
            "--max-cost", "0.50",
            "--test-command", "pytest -q",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["task"] == "fix the bug"
    assert captured["models"] == ["anthropic:claude-sonnet-5", "deepseek:deepseek-chat"]
    assert captured["max_cost"] == 0.50
    assert captured["test_command"] == "pytest -q"


def test_cli_explore_warns_when_max_cost_omitted(project, monkeypatch):
    monkeypatch.chdir(project)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr("mazu.cli.run_explore", lambda *a, **k: [])

    runner = CliRunner()
    result = runner.invoke(main, ["explore", "task", "--approaches", "1"])

    assert "no --max-cost set" in result.output


def test_cli_explore_discloses_allow_shell_forced(project, monkeypatch):
    monkeypatch.chdir(project)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr("mazu.cli.run_explore", lambda *a, **k: [])

    runner = CliRunner()
    result = runner.invoke(main, ["explore", "task", "--approaches", "1"])

    assert "allow-shell is forced on" in result.output
