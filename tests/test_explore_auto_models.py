"""Tests for `mazu explore --auto-models`: fills --models from this project's own
router history (see mazu/runs/router.py), topped up with one model per other
available provider, instead of requiring the user to type out --models by hand.
Never runs a real explore (mazu.cli.run_explore is monkeypatched) -- these only
cover the model-selection logic and its CLI wiring.
"""

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import mazu.cli as cli_module
from mazu.cli import _auto_pick_models, _runs_db_path, _usage_db_path, main
from mazu.runs.store import RunStore
from mazu.usage.store import UsageStore


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "init"], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], check=True)
    (tmp_path / ".mazu").mkdir()


def _seed_history(root: Path, task: str, model: str, n: int, group_prefix: str) -> None:
    run_store = RunStore(_runs_db_path(root))
    usage_store = UsageStore(_usage_db_path())
    for i in range(n):
        run_id = f"{group_prefix}{i}"
        run_store.start(run_id, task, model, 15, 1, True, None, None, False)
        run_store.finish(run_id, status="completed", stop_reason="end_turn")
        run_store.set_explore_outcome(run_id, f"g{group_prefix}{i}", True)
        provider, _, name = model.partition(":")
        usage_store.log("run", run_id, provider, name, 1000, 100, 0.01)
    run_store.close()
    usage_store.close()


def test_auto_pick_models_ranks_by_router_history_first(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _seed_history(tmp_path, "fix the bug in parser.py", "deepseek:deepseek-chat", 3, "d")

    picks = _auto_pick_models(tmp_path, "fix the bug in parser.py", 1)
    assert picks == ["deepseek:deepseek-chat"]


def test_auto_pick_models_tops_up_with_available_providers_not_already_picked(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-not-real")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _seed_history(tmp_path, "fix the bug in parser.py", "deepseek:deepseek-chat", 3, "d")

    picks = _auto_pick_models(tmp_path, "fix the bug in parser.py", 2)
    assert picks[0] == "deepseek:deepseek-chat"
    assert len(picks) == 2
    assert picks[1].startswith("anthropic:")


def test_auto_pick_models_raises_when_not_enough_distinct_models_exist(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(Exception) as exc_info:
        _auto_pick_models(tmp_path, "fix the bug in parser.py", 2)
    assert "could only find" in str(exc_info.value)


def test_cli_rejects_models_and_auto_models_together(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main, ["explore", "fix it", "--models", "anthropic:claude-sonnet-5", "--auto-models"]
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_cli_auto_models_wires_the_picked_list_into_run_explore(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused-not-real")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    captured = {}

    def _fake_run_explore(task, models, **kwargs):
        captured["models"] = models
        return []

    monkeypatch.setattr(cli_module, "run_explore", _fake_run_explore)
    monkeypatch.setattr(cli_module, "format_explore_report", lambda results, test_command: "")

    runner = CliRunner()
    result = runner.invoke(main, ["explore", "fix the bug", "--approaches", "1", "--auto-models"])

    assert result.exit_code == 0, result.output
    assert captured["models"] == ["deepseek:deepseek-chat"]
    assert "[auto-models] picked deepseek:deepseek-chat" in result.output
