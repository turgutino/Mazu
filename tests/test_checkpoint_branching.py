"""Tests for Addendum 5: branching checkpoints (`mazu run --from-checkpoint --branch`,
`mazu checkpoint compare-branches`). Covers the additive index-entry fields
(parent_checkpoint_id, branch), the surgical fixes to the three linear-only
assumptions (prune, timeline_entries, _resolve_entry), CheckpointManager.fork(),
RunStore's lineage columns/migration, UsageStore's session_id filter, and the
run_autonomous wiring that ties a fork into a brand-new run.
"""

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import mazu.agent.autonomous as autonomous_module
from mazu.agent.autonomous import run_autonomous
from mazu.checkpoint.manager import CheckpointManager
from mazu.checkpoint.store import CheckpointIndex
from mazu.cli import main
from mazu.llm.types import AgentResponse
from mazu.runs.store import RunStore
from mazu.tools.registry import ToolRegistry
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
    return tmp_path


def _current_branch(project: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project, capture_output=True, text=True
    ).stdout.strip()


def _end_turn_response(text: str = "done") -> AgentResponse:
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": text}], usage={})


# ---------------------------------------------------------------------------
# parent_checkpoint_id / branch on the entry dict
# ---------------------------------------------------------------------------


def test_snapshot_records_branch_from_git(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")
    assert entry["branch"] == _current_branch(project)


def test_first_checkpoint_of_a_session_has_no_parent(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual", session_id="run-1")
    assert entry["parent_checkpoint_id"] is None


def test_second_checkpoint_of_a_session_points_at_the_first(project: Path):
    manager = CheckpointManager(project)
    first = manager.snapshot(messages=[], trigger="manual", session_id="run-1")
    second = manager.snapshot(messages=[], trigger="manual", session_id="run-1")
    third = manager.snapshot(messages=[], trigger="manual", session_id="run-1")

    assert second["parent_checkpoint_id"] == first["id"]
    assert third["parent_checkpoint_id"] == second["id"]


def test_explicit_parent_checkpoint_id_override_is_respected(project: Path):
    manager = CheckpointManager(project)
    origin = manager.snapshot(messages=[], trigger="manual", session_id="run-1")

    forked_first = manager.snapshot(
        messages=[], trigger="manual", session_id="run-2", parent_checkpoint_id=origin["id"]
    )
    assert forked_first["parent_checkpoint_id"] == origin["id"]


# ---------------------------------------------------------------------------
# CheckpointIndex.last_for_branch
# ---------------------------------------------------------------------------


def test_last_for_branch_scopes_to_the_named_branch(project: Path):
    index = CheckpointIndex(project / ".mazu" / "checkpoints")
    index.append({"id": "cp_000001", "step": 1, "branch": "main"})
    index.append({"id": "cp_000002", "step": 2, "branch": "exp-1"})
    index.append({"id": "cp_000003", "step": 3, "branch": "main"})

    assert index.last_for_branch("main")["id"] == "cp_000003"
    assert index.last_for_branch("exp-1")["id"] == "cp_000002"


def test_last_for_branch_includes_entries_with_no_branch_key(project: Path):
    """Pre-branching entries have no "branch" key at all -- they must still resolve
    on whatever branch they're actually still on, not become invisible."""
    index = CheckpointIndex(project / ".mazu" / "checkpoints")
    index.append({"id": "cp_000001", "step": 1})  # no "branch" key, like pre-addendum history

    assert index.last_for_branch("main")["id"] == "cp_000001"


def test_last_for_branch_no_match_returns_none(project: Path):
    index = CheckpointIndex(project / ".mazu" / "checkpoints")
    index.append({"id": "cp_000001", "step": 1, "branch": "main"})

    assert index.last_for_branch("some-other-branch") is None


# ---------------------------------------------------------------------------
# fork()
# ---------------------------------------------------------------------------


def test_fork_creates_and_checks_out_the_new_branch(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    manager.fork(entry["id"], "exp-1")

    assert _current_branch(project) == "exp-1"


def test_fork_restores_memory_and_skills_onto_the_new_branch(project: Path):
    from mazu.memory.store import MemoryStore
    from mazu.skills.manager import SkillManager

    mazu_dir = project / ".mazu"
    mazu_dir.mkdir()
    store = MemoryStore(mazu_dir / "memory.db")
    store.add(category="fact", title="At fork point", body="x")
    store.close()
    SkillManager(project).save("skill_one", "does a thing", "def run(args):\n    return 'ok'")

    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    # Diverge on the original branch after the checkpoint -- fork must restore the
    # checkpoint's state, not whatever the origin branch has moved on to since.
    store = MemoryStore(mazu_dir / "memory.db")
    store.add(category="fact", title="After checkpoint", body="y")
    store.close()

    manager.fork(entry["id"], "exp-1")

    store = MemoryStore(mazu_dir / "memory.db")
    titles = {row["title"] for row in store.all_active()}
    store.close()
    assert titles == {"At fork point"}
    assert (mazu_dir / "skills" / "skill_one").exists()


def test_fork_returns_the_checkpoints_conversation(project: Path):
    manager = CheckpointManager(project)
    messages = [{"role": "user", "content": "hello"}]
    entry = manager.snapshot(messages=messages, trigger="manual")

    result = manager.fork(entry["id"], "exp-1")

    assert result["messages"] == messages
    assert result["entry"]["id"] == entry["id"]


def test_fork_does_not_truncate_the_origin_branchs_later_history(project: Path):
    """The core difference from restore(): forking is additive divergence, not a
    rollback -- the origin branch's checkpoints made after the fork point must stay
    valid and present in the index."""
    manager = CheckpointManager(project)
    first = manager.snapshot(messages=[], trigger="manual")
    second = manager.snapshot(messages=[], trigger="manual")
    third = manager.snapshot(messages=[], trigger="manual")

    manager.fork(first["id"], "exp-1")

    ids = {e["id"] for e in manager.list_checkpoints()}
    assert {first["id"], second["id"], third["id"]} <= ids


def test_fork_unknown_checkpoint_raises(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    with pytest.raises(ValueError):
        manager.fork("cp_999999", "exp-1")


def test_fork_duplicate_branch_name_raises(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")
    manager.fork(entry["id"], "exp-1")

    # Back on exp-1 now; check out main again before trying to reuse the name so the
    # duplicate-branch error, not a "already on that branch" concern, is what's tested.
    subprocess.run(["git", "checkout", "-"], cwd=project, capture_output=True, text=True)
    with pytest.raises(ValueError):
        manager.fork(entry["id"], "exp-1")


# ---------------------------------------------------------------------------
# prune() branch-safety
# ---------------------------------------------------------------------------


def test_prune_does_not_delete_a_divergent_branchs_only_checkpoint(project: Path):
    manager = CheckpointManager(project, retention=50)
    for _ in range(5):
        manager.snapshot(messages=[], trigger="manual", session_id="main-session")

    entries = manager.list_checkpoints()
    fork_point = entries[1]  # an early checkpoint, still on main at this point
    fork_result = manager.fork(fork_point["id"], "exp-1")
    exp_checkpoint = manager.snapshot(messages=fork_result["messages"], trigger="manual", session_id="exp-session")

    # Go back to main and pile on far more than retention allows.
    subprocess.run(["git", "checkout", "main"], cwd=project, capture_output=True, text=True)
    if _current_branch(project) != "main":
        subprocess.run(["git", "checkout", "master"], cwd=project, capture_output=True, text=True)
    for _ in range(60):
        manager.snapshot(messages=[], trigger="manual", session_id="main-session")

    remaining_ids = {e["id"] for e in manager.list_checkpoints()}
    assert exp_checkpoint["id"] in remaining_ids
    assert (manager.checkpoints_dir / exp_checkpoint["id"]).exists()


def test_prune_still_prunes_within_each_branch_independently(project: Path):
    manager = CheckpointManager(project, retention=3)
    for _ in range(10):
        manager.snapshot(messages=[], trigger="manual")

    remaining = manager.list_checkpoints()
    assert len(remaining) == 3


# ---------------------------------------------------------------------------
# timeline_entries() diffs against the real parent, not the previous list entry
# ---------------------------------------------------------------------------


def test_timeline_diffs_forked_checkpoint_against_its_real_parent(project: Path):
    manager = CheckpointManager(project)
    (project / "a.py").write_text("original")
    origin = manager.snapshot(messages=[], trigger="manual")

    fork_result = manager.fork(origin["id"], "exp-1")
    (project / "b.py").write_text("new on the fork")
    forked_checkpoint = manager.snapshot(
        messages=fork_result["messages"], trigger="manual", session_id="exp-session",
        parent_checkpoint_id=origin["id"],
    )

    timeline = {e["id"]: e for e in manager.timeline_entries()}
    # Must diff against origin's commit (b.py is new), not whatever happens to be
    # adjacent to it in the flat index list.
    assert "b.py" in timeline[forked_checkpoint["id"]]["files_changed"]
    assert "a.py" not in timeline[forked_checkpoint["id"]]["files_changed"]


def test_timeline_falls_back_to_previous_entry_for_pre_branching_data(project: Path):
    """Entries that predate this addendum have no parent_checkpoint_id key at all --
    timeline_entries() must fall back to the old "diff against previous list entry"
    behavior for them, so existing output is unchanged."""
    index = CheckpointIndex(project / ".mazu" / "checkpoints")
    manager = CheckpointManager(project)
    manager.ensure_git_repo()

    (project / "a.py").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=project, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "c1"], cwd=project, capture_output=True, text=True)
    commit1 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True
    ).stdout.strip()
    index.append({"id": "cp_000001", "step": 1, "created_at": "t1", "git_commit": commit1,
                   "trigger": "manual", "summary": "s1"})

    (project / "b.py").write_text("y")
    subprocess.run(["git", "add", "-A"], cwd=project, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "c2"], cwd=project, capture_output=True, text=True)
    commit2 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True
    ).stdout.strip()
    index.append({"id": "cp_000002", "step": 2, "created_at": "t2", "git_commit": commit2,
                   "trigger": "manual", "summary": "s2"})

    timeline = manager.timeline_entries()
    assert timeline[0]["files_changed"] == []
    assert "b.py" in timeline[1]["files_changed"]


# ---------------------------------------------------------------------------
# _resolve_entry(None) targets the current branch's own most recent checkpoint
# ---------------------------------------------------------------------------


def test_rollback_with_no_argument_targets_the_current_branchs_own_checkpoint(project: Path):
    manager = CheckpointManager(project)
    origin = manager.snapshot(messages=[], trigger="manual")
    manager.fork(origin["id"], "exp-1")
    exp_checkpoint = manager.snapshot(messages=[], trigger="manual")  # made on exp-1

    entry, _ = manager.preview_rollback(None)
    assert entry["id"] == exp_checkpoint["id"]

    subprocess.run(["git", "checkout", "main"], cwd=project, capture_output=True, text=True)
    if _current_branch(project) != "main":
        subprocess.run(["git", "checkout", "master"], cwd=project, capture_output=True, text=True)

    entry, _ = manager.preview_rollback(None)
    assert entry["id"] == origin["id"]


# ---------------------------------------------------------------------------
# RunStore lineage columns + migration
# ---------------------------------------------------------------------------


def test_run_store_start_accepts_lineage_kwargs(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.start(
        "fork-run", "divergent task", "deepseek:deepseek-chat", 15, 1, False, None, None, False,
        origin_checkpoint_id="cp_000002", parent_run_id="original-run", branch_name="exp-1",
    )

    row = store.get("fork-run")
    assert row["origin_checkpoint_id"] == "cp_000002"
    assert row["parent_run_id"] == "original-run"
    assert row["branch_name"] == "exp-1"


def test_run_store_start_without_lineage_kwargs_defaults_to_null(tmp_path):
    """Every pre-addendum call site (plain runs, --resume) passes nothing new --
    must not break or require the new kwargs."""
    store = RunStore(tmp_path / "runs.db")
    store.start("r1", "task", "deepseek:deepseek-chat", 15, 1, False, None, None, False)

    row = store.get("r1")
    assert row["origin_checkpoint_id"] is None
    assert row["parent_run_id"] is None
    assert row["branch_name"] is None


def test_run_store_migrates_lineage_columns_onto_a_pre_existing_db(tmp_path):
    """Simulates a runs.db created before this addendum (no lineage columns) --
    opening it with the new RunStore must not crash, and must add the columns."""
    import sqlite3

    db_path = tmp_path / "runs.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY, task TEXT NOT NULL, model TEXT, max_steps INTEGER NOT NULL,
            checkpoint_every INTEGER NOT NULL, allow_shell INTEGER NOT NULL,
            shell_allowlist TEXT, max_cost REAL, dry_run INTEGER NOT NULL,
            status TEXT NOT NULL, stop_reason TEXT, started_at TEXT NOT NULL, ended_at TEXT,
            last_step INTEGER NOT NULL DEFAULT 0, last_checkpoint_id TEXT,
            checkpoints_created INTEGER NOT NULL DEFAULT 0, memories_saved INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        "INSERT INTO runs (id, task, max_steps, checkpoint_every, allow_shell, dry_run, "
        "status, started_at) VALUES ('old-run', 'legacy task', 15, 1, 0, 0, 'completed', 't1')"
    )
    conn.commit()
    conn.close()

    store = RunStore(db_path)  # must not raise
    row = store.get("old-run")
    assert row["task"] == "legacy task"
    assert row["origin_checkpoint_id"] is None  # migrated column, NULL for pre-existing rows


# ---------------------------------------------------------------------------
# UsageStore.summary() session_id filter
# ---------------------------------------------------------------------------


def test_usage_summary_session_id_filter_scopes_to_one_run(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    store.log("run", "run-a", "anthropic", "claude-sonnet-5", 100, 50, 0.01)
    store.log("run", "run-b", "anthropic", "claude-sonnet-5", 200, 100, 0.02)

    summary_a = store.summary(session_id="run-a")
    assert summary_a["total_calls"] == 1
    assert summary_a["total_cost"] == pytest.approx(0.01)

    summary_all = store.summary()
    assert summary_all["total_calls"] == 2


def test_usage_summary_unknown_session_id_is_empty(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    store.log("run", "run-a", "anthropic", "claude-sonnet-5", 100, 50, 0.01)

    summary = store.summary(session_id="no-such-run")
    assert summary["total_calls"] == 0
    assert summary["total_cost"] == 0.0


# ---------------------------------------------------------------------------
# run_autonomous fork wiring
# ---------------------------------------------------------------------------


def test_fork_run_starts_a_new_session_and_run_store_row(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    run_store = RunStore(tmp_path / "runs.db")

    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    run_autonomous(
        registry=ToolRegistry(),
        task="a different, divergent task",
        session_id="fork-run",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
        model="deepseek:deepseek-chat",
        run_store=run_store,
        resume_messages=[{"role": "user", "content": "original task"}],
        origin_checkpoint_id="cp_000002",
        parent_run_id="original-run",
        branch_name="exp-1",
    )

    # run_autonomous closes run_store itself in its finally block -- reopen against
    # the same file to inspect the persisted row, matching test_run_report_resume.py.
    row = RunStore(tmp_path / "runs.db").get("fork-run")
    assert row is not None  # is_fork must NOT skip run_store.start() the way true resume does
    assert row["origin_checkpoint_id"] == "cp_000002"
    assert row["parent_run_id"] == "original-run"
    assert row["branch_name"] == "exp-1"


def test_fork_run_appends_the_new_task_after_the_seeded_history(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    captured_messages = []

    def _fake_run_turn(messages, *a, **k):
        captured_messages.append(list(messages))
        return _end_turn_response()

    monkeypatch.setattr(autonomous_module, "run_turn", _fake_run_turn)

    prior = [{"role": "user", "content": "original task"}]
    run_autonomous(
        registry=ToolRegistry(),
        task="the new divergent task",
        session_id="fork-run",
        checkpoint_manager=checkpoint_manager,
        max_steps=1,
        model="deepseek:deepseek-chat",
        resume_messages=prior,
        origin_checkpoint_id="cp_000002",
        branch_name="exp-1",
    )

    # Unlike a true resume (which reuses resume_messages as-is), a fork must append
    # the new task on top of the seeded history -- that's the whole point of forking
    # with a *different* task rather than resuming the same one.
    first_call = captured_messages[0]
    assert len(first_call) == 2
    assert first_call[-1]["content"] == "the new divergent task"


def test_fork_first_checkpoint_gets_the_origin_as_parent(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    from mazu.tools.fs import make_fs_tools
    from mazu.llm.types import AgentResponse

    registry = ToolRegistry()
    for tool in make_fs_tools(tmp_path):
        registry.register(tool)

    responses = iter(
        [
            AgentResponse(
                stop_reason="tool_use",
                content=[{"type": "tool_use", "id": "t1", "name": "write_file",
                           "input": {"path": "a.py", "content": "x"}}],
                usage={},
            ),
            _end_turn_response(),
        ]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(responses))

    run_autonomous(
        registry=registry,
        task="new task",
        session_id="fork-run",
        checkpoint_manager=checkpoint_manager,
        max_steps=5,
        checkpoint_every=1,
        model="deepseek:deepseek-chat",
        resume_messages=[{"role": "user", "content": "original task"}],
        origin_checkpoint_id="cp_000099",
    )

    first_checkpoint = checkpoint_manager.latest_for_session("fork-run")
    assert first_checkpoint["parent_checkpoint_id"] == "cp_000099"


# ---------------------------------------------------------------------------
# CLI: mazu run --from-checkpoint / --branch
# ---------------------------------------------------------------------------


def test_from_checkpoint_and_resume_are_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--from-checkpoint", "cp_000001", "--branch", "exp-1", "--resume", "r1", "task"]
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_from_checkpoint_requires_branch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--from-checkpoint", "cp_000001", "task"])
    assert result.exit_code != 0
    assert "--branch" in result.output


def test_branch_without_from_checkpoint_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--branch", "exp-1", "task"])
    assert result.exit_code != 0
    assert "--from-checkpoint" in result.output


def test_from_checkpoint_requires_a_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--from-checkpoint", "cp_000001", "--branch", "exp-1"])
    assert result.exit_code != 0
    assert "TASK" in result.output


def test_from_checkpoint_unknown_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--from-checkpoint", "cp_999999", "--branch", "exp-1", "task"]
    )
    assert result.exit_code == 0  # handled gracefully, not a crash/traceback
    assert "No checkpoint found" in result.output


def test_from_checkpoint_end_to_end_forks_and_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    runner = CliRunner()

    # Seed an origin checkpoint via a real (mocked) `mazu run` -- needs at least one
    # tool_use round before end_turn, since a checkpoint is only taken after a round
    # of tool execution (an immediate end_turn produces zero checkpoints).
    seed_responses = iter(
        [
            AgentResponse(
                stop_reason="tool_use",
                content=[{"type": "tool_use", "id": "t1", "name": "write_file",
                           "input": {"path": "seed.py", "content": "x"}}],
                usage={},
            ),
            _end_turn_response(),
        ]
    )
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: next(seed_responses))
    result = runner.invoke(main, ["run", "original task"])
    assert result.exit_code == 0, result.output

    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    entries = checkpoint_manager.list_checkpoints()
    assert entries, "the seeded run should have produced at least one checkpoint"
    origin_id = entries[0]["id"]

    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())
    result = runner.invoke(
        main, ["run", "--from-checkpoint", origin_id, "--branch", "exp-1", "a divergent task"]
    )
    assert result.exit_code == 0, result.output
    assert "Forked from" in result.output
    assert _current_branch(root) == "exp-1"


# ---------------------------------------------------------------------------
# CLI: mazu checkpoint compare-branches
# ---------------------------------------------------------------------------


def test_compare_branches_unknown_run_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["checkpoint", "compare-branches", "no-such-a", "no-such-b"])
    assert result.exit_code == 0
    assert "No run found" in result.output


def test_compare_branches_shows_both_runs_and_diff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    (root / ".gitignore").write_text(".mazu/\n", encoding="utf-8")

    checkpoint_manager = CheckpointManager(root)
    (root / "a.py").write_text("from run a")
    entry_a = checkpoint_manager.snapshot(messages=[], trigger="manual", session_id="run-a")

    (root / "b.py").write_text("from run b")
    entry_b = checkpoint_manager.snapshot(messages=[], trigger="manual", session_id="run-b")

    run_store = RunStore(root / ".mazu" / "runs.db")
    run_store.start("run-a", "task a", "deepseek:deepseek-chat", 15, 1, False, None, None, False)
    run_store.update_progress("run-a", 1, checkpoint_id=entry_a["id"])
    run_store.finish("run-a", "completed", "end_turn", memories_saved=1)
    run_store.start("run-b", "task b", "deepseek:deepseek-chat", 15, 1, False, None, None, False)
    run_store.update_progress("run-b", 1, checkpoint_id=entry_b["id"])
    run_store.finish("run-b", "completed", "end_turn", memories_saved=2)
    run_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["checkpoint", "compare-branches", "run-a", "run-b"])

    assert result.exit_code == 0, result.output
    assert "run-a" in result.output
    assert "run-b" in result.output
    assert "Memories saved:    1" in result.output
    assert "Memories saved:    2" in result.output
    assert "b.py" in result.output


# ---------------------------------------------------------------------------
# Regression: existing single-chain behavior is unchanged
# ---------------------------------------------------------------------------


def test_regression_timeline_still_works_for_a_plain_linear_history(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")
    (project / "new_file.py").write_text("x")
    manager.snapshot(messages=[], trigger="manual")

    timeline = manager.timeline_entries()
    assert timeline[0]["files_changed"] == []
    assert "new_file.py" in timeline[1]["files_changed"]


def test_regression_rollback_with_no_argument_still_targets_most_recent(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")
    second = manager.snapshot(messages=[{"role": "user", "content": "x"}], trigger="manual")

    shown = manager.show_entry()
    assert shown["id"] == second["id"]
