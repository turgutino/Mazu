"""Verifies run_autonomous's own integration with the new `shared_cost_tracker`
param (added for `mazu explore`'s cross-branch budget): each step's cost is fed
into the tracker, and a shared-budget exhaustion (even one caused entirely by a
SIBLING branch, not this run's own local max_cost) stops the run with a distinct
stop_reason. Complements test_explore.py's higher-level "tracker is genuinely
shared across two real threads" test with a direct, single-run proof of the
wiring itself.
"""

import subprocess

import pytest

import mazu.agent.autonomous as autonomous_module
from mazu.agent.autonomous import run_autonomous
from mazu.agent.council import _SharedCostTracker
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.types import AgentResponse
from mazu.runs.store import RunStore
from mazu.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def _tool_use_response() -> AgentResponse:
    return AgentResponse(
        stop_reason="tool_use",
        content=[{"type": "tool_use", "id": "t1", "name": "list_dir", "input": {}}],
        usage={"input_tokens": 1000, "output_tokens": 500},
    )


def test_shared_tracker_receives_this_runs_cost(tmp_path, monkeypatch):
    checkpoint_manager = CheckpointManager(tmp_path)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _tool_use_response())
    tracker = _SharedCostTracker(max_cost=None)

    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s1",
        checkpoint_manager=checkpoint_manager,
        model="anthropic:claude-sonnet-5",
        max_steps=1,
        shared_cost_tracker=tracker,
    )

    assert tracker.total > 0


def test_run_stops_when_a_sibling_already_exhausted_the_shared_budget(tmp_path, monkeypatch):
    """The core scenario mazu explore needs: this run has no local --max-cost of
    its own, but a SIBLING branch already spent the whole shared budget before
    this run's very first step even reports its cost -- the shared check must
    still catch it.
    """
    checkpoint_manager = CheckpointManager(tmp_path)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _tool_use_response())

    tracker = _SharedCostTracker(max_cost=0.001)
    tracker.add_and_check(1.0)  # simulate a sibling branch already blowing the budget
    assert tracker.is_exhausted() is True

    runs_db = tmp_path / ".mazu" / "runs.db"
    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s2",
        checkpoint_manager=checkpoint_manager,
        model="anthropic:claude-sonnet-5",
        max_steps=15,
        shared_cost_tracker=tracker,
        run_store=RunStore(runs_db),
    )

    # run_autonomous's own `finally` block already closed the RunStore instance
    # passed in above -- open a fresh connection to read the row back, same
    # pattern explore.py itself uses.
    result_store = RunStore(runs_db)
    row = result_store.get("s2")
    assert row["stop_reason"] == "shared_max_cost"
    # Stopped on step 1, never reached anywhere close to max_steps=15.
    assert row["last_step"] <= 1
    result_store.close()


def test_local_max_cost_still_works_with_no_shared_tracker(tmp_path, monkeypatch):
    """Regression: passing shared_cost_tracker=None (every existing caller --
    mazu run, mazu chat, mazu council -- never sets it) must leave the existing
    per-run --max-cost behavior completely unchanged.
    """
    checkpoint_manager = CheckpointManager(tmp_path)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _tool_use_response())

    runs_db = tmp_path / ".mazu" / "runs.db"
    run_autonomous(
        registry=ToolRegistry(),
        task="do something",
        session_id="s3",
        checkpoint_manager=checkpoint_manager,
        model="anthropic:claude-sonnet-5",
        max_steps=15,
        max_cost=0.0001,  # exhausted almost immediately by local cost alone
        shared_cost_tracker=None,
        run_store=RunStore(runs_db),
    )
    result_store = RunStore(runs_db)
    row = result_store.get("s3")
    assert row["stop_reason"] == "max_cost"
    result_store.close()
