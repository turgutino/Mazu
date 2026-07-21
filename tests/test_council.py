"""Tests for Addendum 6's council mode cost guardrail: a thread-safe shared cost
tracker that stops members from taking further rounds once a --max-cost budget is
exhausted, and skips the lead synthesis call entirely if the budget was already used
up by the member round(s).
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from mazu.agent.council import _SharedCostTracker, run_council
from mazu.llm.types import AgentResponse
from mazu.tools.registry import ToolRegistry


def _end_turn_response(text: str = "done") -> AgentResponse:
    return AgentResponse(
        stop_reason="end_turn",
        content=[{"type": "text", "text": text}],
        usage={"input_tokens": 100, "output_tokens": 50},
    )


def _tool_use_response(text: str = "") -> AgentResponse:
    content = [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}}]
    if text:
        content.insert(0, {"type": "text", "text": text})
    return AgentResponse(stop_reason="tool_use", content=content, usage={"input_tokens": 1000, "output_tokens": 500})


# ---------------------------------------------------------------------------
# _SharedCostTracker: thread-safety and basic semantics
# ---------------------------------------------------------------------------


def test_shared_cost_tracker_none_budget_is_always_a_noop():
    tracker = _SharedCostTracker(max_cost=None)
    assert tracker.add_and_check(1000.0) is False
    assert tracker.is_exhausted() is False
    assert tracker.total == 1000.0


def test_shared_cost_tracker_flips_exhausted_at_the_right_point():
    tracker = _SharedCostTracker(max_cost=0.10)
    assert tracker.add_and_check(0.05) is False
    assert tracker.add_and_check(0.04) is False
    assert tracker.add_and_check(0.02) is True  # total is now 0.11 >= 0.10
    assert tracker.is_exhausted() is True


def test_shared_cost_tracker_none_cost_adds_nothing_but_reports_current_state():
    tracker = _SharedCostTracker(max_cost=0.01)
    tracker.add_and_check(0.02)  # exhausts the budget
    # An untrackable model's cost (None) doesn't clear the exhausted state or crash.
    assert tracker.add_and_check(None) is True
    assert tracker.total == 0.02


def test_shared_cost_tracker_no_lost_updates_under_concurrency():
    """The concrete proof the lock actually works: hammer the same tracker with many
    small increments from real threads and confirm the total is exact -- a race
    (read-then-add without a lock) would silently drop some of these additions.
    """
    tracker = _SharedCostTracker(max_cost=None)
    increments = [0.0001] * 2000

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(tracker.add_and_check, increments))

    # Floating point sums of many small increments won't be bit-exact regardless of
    # the lock -- pytest.approx accounts for that, while still proving no update was
    # silently dropped (a lost update would be off by whole increments, not ULPs).
    assert tracker.total == pytest.approx(sum(increments))


# ---------------------------------------------------------------------------
# run_council: --max-cost cuts members short and skips the lead call
# ---------------------------------------------------------------------------


def test_council_without_max_cost_behaves_exactly_as_before(monkeypatch):
    def _fake_run_turn(messages, system, tools, model=None):
        return _end_turn_response(f"answer from {model}")

    with patch("mazu.agent.council.run_turn", side_effect=_fake_run_turn):
        result = run_council(
            "question",
            models=["anthropic:claude-sonnet-5", "anthropic:claude-haiku-4-5"],
            lead_model="anthropic:claude-sonnet-5",
            full_registry=ToolRegistry(),
        )

    assert result == "answer from anthropic:claude-sonnet-5"  # the lead call's own answer
    assert "Skipped lead synthesis" not in result


def test_council_max_cost_skips_lead_synthesis_when_already_exhausted(monkeypatch):
    # Members always return tool_use (so they'd keep looping) with a large enough
    # usage that even ONE round per member blows straight through a tiny budget.
    def _fake_run_turn(messages, system, tools, model=None):
        return _tool_use_response("thinking")

    with patch("mazu.agent.council.run_turn", side_effect=_fake_run_turn), patch(
        "mazu.agent.council.record_action", lambda *a, **k: None
    ):
        result = run_council(
            "question",
            models=["anthropic:claude-sonnet-5", "anthropic:claude-haiku-4-5"],
            lead_model="anthropic:claude-sonnet-5",
            full_registry=ToolRegistry(),
            max_cost=0.0001,  # deliberately tiny -- exhausted after the first round
        )

    assert "Skipped lead synthesis" in result
    assert "--max-cost budget" in result


def test_council_max_cost_counts_members_that_never_use_a_tool(monkeypatch):
    """Regression test for a real bug found via live testing: a member that answers
    directly on its first round (stop_reason == "end_turn", never touches a tool)
    used to skip the cost-tracking code entirely, because the check was placed AFTER
    the "if stop_reason != tool_use: break" early-exit -- so a --max-cost budget was
    silently never enforced for the common case of simple, no-tool-needed questions.
    This reproduces exactly that scenario (both members answer immediately) and
    confirms their cost now DOES count toward the shared budget.
    """

    def _fake_run_turn(messages, system, tools, model=None):
        return _end_turn_response("Paris")  # answers immediately, no tool_use round

    with patch("mazu.agent.council.run_turn", side_effect=_fake_run_turn):
        result = run_council(
            "What is the capital of France?",
            models=["anthropic:claude-haiku-4-5", "anthropic:claude-sonnet-5"],
            lead_model="anthropic:claude-opus-4-8",
            full_registry=ToolRegistry(),
            max_cost=0.0000001,  # smaller than even one no-tool-use round's cost
        )

    assert "Skipped lead synthesis" in result


def test_council_max_cost_none_never_skips_lead(monkeypatch):
    def _fake_run_turn(messages, system, tools, model=None):
        return _end_turn_response("answer")

    with patch("mazu.agent.council.run_turn", side_effect=_fake_run_turn):
        result = run_council(
            "question",
            models=["anthropic:claude-sonnet-5"],
            lead_model="anthropic:claude-sonnet-5",
            full_registry=ToolRegistry(),
            max_cost=None,
        )

    assert "Skipped lead synthesis" not in result


def test_council_always_prints_cost_awareness_line(monkeypatch, capsys):
    def _fake_run_turn(messages, system, tools, model=None):
        return _end_turn_response("answer")

    with patch("mazu.agent.council.run_turn", side_effect=_fake_run_turn):
        run_council(
            "question",
            models=["anthropic:claude-sonnet-5", "anthropic:claude-haiku-4-5"],
            lead_model="anthropic:claude-sonnet-5",
            full_registry=ToolRegistry(),
        )

    out = capsys.readouterr().out
    assert "[cost] Council mode queries 2 models" in out


# ---------------------------------------------------------------------------
# CLI wiring: mazu council --max-cost
# ---------------------------------------------------------------------------


def test_cli_council_max_cost_option_reaches_run_council(tmp_path, monkeypatch):
    import subprocess

    from click.testing import CliRunner

    from mazu.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-not-real")
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])

    captured = {}

    def _fake_run_council(question, models, lead_model, full_registry, **kwargs):
        captured["max_cost"] = kwargs.get("max_cost")
        return "ok"

    monkeypatch.setattr("mazu.cli.run_council", _fake_run_council)

    runner = CliRunner()
    result = runner.invoke(main, ["council", "a question", "--max-cost", "0.05"])

    assert result.exit_code == 0, result.output
    assert captured["max_cost"] == 0.05
