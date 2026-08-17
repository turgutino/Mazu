"""CLI wiring for the Learning Model Router: the passive suggestion print in
`mazu run` (and the config opt-out), plus `mazu router stats`. Mocks at the
`run_turn` level (not run_autonomous) so the real CLI code path -- including
_print_router_suggestion -- executes for real, matching
test_checkpoint_branching.py::test_from_checkpoint_end_to_end_forks_and_runs's
established pattern for full end-to-end `mazu run` CLI tests.
"""

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import mazu.agent.autonomous as autonomous_module
import mazu.agent.loop as loop_module
from mazu.agent.loop import run_chat_loop
from mazu.checkpoint.manager import CheckpointManager
from mazu.cli import _runs_db_path, _usage_db_path, main
from mazu.llm.types import AgentResponse
from mazu.runs.store import RunStore
from mazu.tools.registry import ToolRegistry
from mazu.usage.store import UsageStore


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def _end_turn_response(text: str = "done") -> AgentResponse:
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": text}], usage={})


def _seed_explore_history(root: Path, n: int = 3) -> None:
    """Seeds enough real explore-branch history (>= MIN_SAMPLES_FOR_SUGGESTION) for
    'fix the bug in parser.py' to trigger a real suggestion, directly against the
    project's real .mazu/runs.db and ~/.mazu/usage.db -- exactly what a real `mazu
    explore` run would have left behind, without actually running one.
    """
    run_store = RunStore(_runs_db_path(root))
    usage_store = UsageStore(_usage_db_path())
    for i in range(n):
        run_store.start(f"cheap{i}", "fix the bug in parser.py", "deepseek:deepseek-chat", 15, 1, True, None, None, False)
        run_store.finish(f"cheap{i}", status="completed", stop_reason="end_turn")
        run_store.set_explore_outcome(f"cheap{i}", f"g{i}", None)
        usage_store.log("run", f"cheap{i}", "deepseek", "deepseek-chat", 1000, 100, 0.01)

        run_store.start(f"pricey{i}", "fix the bug in parser.py", "anthropic:claude-sonnet-5", 15, 1, True, None, None, False)
        run_store.finish(f"pricey{i}", status="completed", stop_reason="end_turn")
        run_store.set_explore_outcome(f"pricey{i}", f"g{i}", None)
        usage_store.log("run", f"pricey{i}", "anthropic", "claude-sonnet-5", 1000, 100, 0.10)
    run_store.close()
    usage_store.close()


# ---------------------------------------------------------------------------
# mazu run -- suggestion print
# ---------------------------------------------------------------------------


def test_run_without_model_prints_suggestion_when_history_exists(tmp_path, monkeypatch):
    _seed_explore_history(tmp_path)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    runner = CliRunner()
    result = runner.invoke(main, ["run", "fix the bug in parser.py"])

    assert result.exit_code == 0, result.output
    assert "[router]" in result.output
    assert "deepseek:deepseek-chat" in result.output
    assert "never applied automatically" in result.output


def test_run_with_explicit_model_never_prints_suggestion(tmp_path, monkeypatch):
    _seed_explore_history(tmp_path)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--model", "anthropic:claude-sonnet-5", "fix the bug in parser.py"])

    assert result.exit_code == 0, result.output
    assert "[router]" not in result.output


def test_run_prints_no_suggestion_below_min_sample_threshold(tmp_path, monkeypatch):
    _seed_explore_history(tmp_path, n=1)  # below MIN_SAMPLES_FOR_SUGGESTION
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    runner = CliRunner()
    result = runner.invoke(main, ["run", "fix the bug in parser.py"])

    assert result.exit_code == 0, result.output
    assert "[router]" not in result.output


def test_run_prints_no_suggestion_with_zero_history(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    runner = CliRunner()
    result = runner.invoke(main, ["run", "fix the bug in parser.py"])

    assert result.exit_code == 0, result.output
    assert "[router]" not in result.output


def test_router_suggestions_false_config_silences_the_print(tmp_path, monkeypatch):
    _seed_explore_history(tmp_path)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    runner = CliRunner()
    config_result = runner.invoke(main, ["config", "set", "router_suggestions", "false"])
    assert config_result.exit_code == 0, config_result.output

    result = runner.invoke(main, ["run", "fix the bug in parser.py"])

    assert result.exit_code == 0, result.output
    assert "[router]" not in result.output


# ---------------------------------------------------------------------------
# Critical correctness gate: the suggestion must NEVER change which model runs
# ---------------------------------------------------------------------------


def test_router_suggestion_never_changes_model_passed_to_run_autonomous(tmp_path, monkeypatch):
    """The single most important test in this addendum: even with real, strong
    seeded history pointing at deepseek, and no --model given, the model actually
    passed into run_autonomous() must still be None (letting default_model()'s own
    ordinary resolution decide) -- never silently overridden to the suggested model.
    """
    _seed_explore_history(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused-not-real")
    captured = {}
    real_run_autonomous = autonomous_module.run_autonomous

    def _capture(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        return real_run_autonomous(*args, **kwargs)

    monkeypatch.setattr("mazu.cli.run_autonomous", _capture)
    monkeypatch.setattr(autonomous_module, "run_turn", lambda *a, **k: _end_turn_response())

    runner = CliRunner()
    result = runner.invoke(main, ["run", "fix the bug in parser.py"])

    assert result.exit_code == 0, result.output
    assert "[router]" in result.output  # the suggestion really did fire
    assert captured["model"] is None  # ...but never mutated what's actually used


# ---------------------------------------------------------------------------
# mazu router stats
# ---------------------------------------------------------------------------


def test_router_stats_shows_no_history_message_when_empty():
    runner = CliRunner()
    result = runner.invoke(main, ["router", "stats"])
    assert result.exit_code == 0, result.output
    assert "No explore history" in result.output


def test_router_stats_plain_text_shows_win_rate_and_cost(tmp_path):
    _seed_explore_history(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["router", "stats"])
    assert result.exit_code == 0, result.output
    assert "deepseek:deepseek-chat" in result.output
    assert "wins" in result.output


def test_router_stats_json_output(tmp_path):
    _seed_explore_history(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["router", "stats", "--json"])
    assert result.exit_code == 0, result.output

    import json

    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    models = {row["model"] for row in payload["data"]}
    assert "deepseek:deepseek-chat" in models


def test_router_stats_task_type_filter(tmp_path):
    _seed_explore_history(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["router", "stats", "--task-type", "feature"])
    assert result.exit_code == 0, result.output
    assert "No explore history recorded yet for task type 'feature'" in result.output


# ---------------------------------------------------------------------------
# mazu chat -- deferred suggestion on the first user message
# ---------------------------------------------------------------------------


def _end_turn_stream(messages, system, tools, on_delta, model=None) -> AgentResponse:
    return _end_turn_response()


def test_chat_prints_suggestion_on_first_message_when_history_exists(tmp_path, monkeypatch, capsys):
    _seed_explore_history(tmp_path)
    checkpoint_manager = CheckpointManager(tmp_path)
    registry = ToolRegistry()

    monkeypatch.setattr(loop_module, "run_turn_stream", _end_turn_stream)
    inputs = iter(["fix the bug in parser.py", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry,
            session_id="s1",
            checkpoint_manager=checkpoint_manager,
            model=None,  # no --model passed, matching the suggestion's guard condition
        )

    out = capsys.readouterr().out
    assert "[router]" in out
    assert "deepseek:deepseek-chat" in out


def test_chat_with_explicit_model_never_prints_suggestion(tmp_path, monkeypatch, capsys):
    _seed_explore_history(tmp_path)
    checkpoint_manager = CheckpointManager(tmp_path)
    registry = ToolRegistry()

    monkeypatch.setattr(loop_module, "run_turn_stream", _end_turn_stream)
    inputs = iter(["fix the bug in parser.py", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    with pytest.raises(StopIteration):
        run_chat_loop(
            registry,
            session_id="s1",
            checkpoint_manager=checkpoint_manager,
            model="anthropic:claude-sonnet-5",
        )

    assert "[router]" not in capsys.readouterr().out
