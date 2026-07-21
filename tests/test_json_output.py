"""Tests for Addendum 7: --json output on the five read-oriented commands
(timeline, memory list, log/log show, runs, models). Uses Click's CliRunner against
an isolated filesystem, exercising the real command wiring end-to-end.
"""

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import mazu
from mazu.checkpoint.manager import CheckpointManager
from mazu.checkpoint.store import CheckpointIndex
from mazu.cli import main
from mazu.memory.store import MemoryStore
from mazu.output import SCHEMA_VERSION
from mazu.runs.store import RunStore


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def _invoke_json(runner, args):
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _assert_envelope(payload, data_type=list):
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["mazu_version"] == mazu.__version__
    assert isinstance(payload["data"], data_type)
    return payload["data"]


# ---------------------------------------------------------------------------
# Envelope shape + empty-result behavior, across all five commands
# ---------------------------------------------------------------------------


def test_timeline_json_envelope_on_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    payload = _invoke_json(runner, ["timeline", "--json"])
    assert _assert_envelope(payload) == []


def test_timeline_json_matches_underlying_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    entry = manager.snapshot(messages=[], trigger="manual")

    runner = CliRunner()
    payload = _invoke_json(runner, ["timeline", "--json"])
    data = _assert_envelope(payload)
    assert len(data) == 1
    assert data[0]["id"] == entry["id"]
    assert data[0]["git_commit"] == entry["git_commit"]
    assert "files_changed" in data[0]


def test_timeline_json_handles_pre_branching_entry_with_absent_keys(tmp_path, monkeypatch):
    """Entries recorded before branching support existed lack branch/
    parent_checkpoint_id keys entirely -- --json must not crash on absence."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    manager.ensure_git_repo()

    (tmp_path / "a.py").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "c1"], cwd=tmp_path, capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()
    index = CheckpointIndex(tmp_path / ".mazu" / "checkpoints")
    index.append(
        {"id": "cp_000001", "step": 1, "created_at": "t1", "git_commit": commit,
         "trigger": "manual", "summary": "s1"}
    )

    runner = CliRunner()
    payload = _invoke_json(runner, ["timeline", "--json"])
    data = _assert_envelope(payload)
    assert data[0]["id"] == "cp_000001"
    assert "branch" not in data[0]
    assert "parent_checkpoint_id" not in data[0]


def test_memory_list_json_envelope_on_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    payload = _invoke_json(runner, ["memory", "list", "--json"])
    assert _assert_envelope(payload) == []


def test_memory_list_json_pinned_is_real_bool_and_no_embedding_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mazu_dir = tmp_path / ".mazu"
    mazu_dir.mkdir()
    store = MemoryStore(mazu_dir / "memory.db")
    mid = store.add(category="fact", title="Pinned fact", body="x", source="explicit")
    store.pin(mid)
    store.close()

    runner = CliRunner()
    payload = _invoke_json(runner, ["memory", "list", "--json"])
    data = _assert_envelope(payload)
    assert len(data) == 1
    assert data[0]["pinned"] is True
    assert "embedding" not in data[0]


def test_memory_list_json_respects_global_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    from mazu.cli import _global_memory_db_path

    global_store = MemoryStore(_global_memory_db_path())
    global_store.add(category="user_preference", title="Global fact", body="y", source="explicit")
    global_store.close()

    runner = CliRunner()
    payload = _invoke_json(runner, ["memory", "list", "--global", "--json"])
    data = _assert_envelope(payload)
    assert len(data) == 1
    assert data[0]["title"] == "Global fact"

    # Project store (no --global) is still empty -- confirms --json didn't bypass
    # the existing store-selection logic.
    payload = _invoke_json(runner, ["memory", "list", "--json"])
    assert _assert_envelope(payload) == []


def test_runs_json_envelope_on_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    payload = _invoke_json(runner, ["runs", "--json"])
    assert _assert_envelope(payload) == []


def test_runs_json_bool_casts_and_shell_allowlist_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mazu_dir = tmp_path / ".mazu"
    mazu_dir.mkdir()
    store = RunStore(mazu_dir / "runs.db")
    store.start(
        "run-a", "task a", "deepseek:deepseek-chat", 15, 1, True, ["git", "npm"], None, True,
        origin_checkpoint_id="cp_000002", parent_run_id="run-main", branch_name="exp-1",
    )
    store.start("run-b", "task b", "deepseek:deepseek-chat", 15, 1, False, None, None, False)
    store.close()

    runner = CliRunner()
    payload = _invoke_json(runner, ["runs", "--json"])
    data = _assert_envelope(payload)
    by_id = {r["id"]: r for r in data}

    assert by_id["run-a"]["allow_shell"] is True
    assert by_id["run-a"]["dry_run"] is True
    assert by_id["run-a"]["shell_allowlist"] == ["git", "npm"]
    assert by_id["run-a"]["origin_checkpoint_id"] == "cp_000002"
    assert by_id["run-a"]["parent_run_id"] == "run-main"
    assert by_id["run-a"]["branch_name"] == "exp-1"

    assert by_id["run-b"]["allow_shell"] is False
    assert by_id["run-b"]["dry_run"] is False
    assert by_id["run-b"]["shell_allowlist"] == []
    assert by_id["run-b"]["origin_checkpoint_id"] is None


def test_log_json_envelope_on_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    payload = _invoke_json(runner, ["log", "--json"])
    assert _assert_envelope(payload) == []


def test_log_show_json_envelope_for_unknown_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    payload = _invoke_json(runner, ["log", "--json", "show", "no-such-session"])
    assert _assert_envelope(payload) == []


def test_log_show_json_flag_must_precede_subcommand(tmp_path, monkeypatch):
    """--json is a group-level option -- `mazu log show <id> --json` (flag AFTER the
    subcommand) is expected to fail per Click's own group-option mechanics, not a bug.
    """
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["log", "show", "some-id", "--json"])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_log_and_log_show_json_with_real_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from mazu.action_log.store import ActionLogStore, record_action

    mazu_dir = tmp_path / ".mazu"
    mazu_dir.mkdir()
    store = ActionLogStore(mazu_dir / "action_log.db")
    record_action(store, "session-1", "run", "write_file", {"path": "a.py"}, "ok", "wrote 10 bytes")
    store.close()

    runner = CliRunner()
    payload = _invoke_json(runner, ["log", "--json"])
    sessions = _assert_envelope(payload)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "session-1"

    payload = _invoke_json(runner, ["log", "--json", "show", "session-1"])
    actions = _assert_envelope(payload)
    assert len(actions) == 1
    assert actions[0]["tool_name"] == "write_file"
    assert actions[0]["changed_file"] == "a.py"


def test_models_json_shape_and_note(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["models", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    data = _assert_envelope(payload, data_type=dict)
    assert "models" in data
    assert "note" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    assert "anthropic:claude-sonnet-5" in {f"{m['provider']}:{m['model']}" for m in data["models"]}
    sonnet = next(m for m in data["models"] if m["model"] == "claude-sonnet-5")
    assert sonnet["input_price_per_million"] == 3.0

    # models --json output text and the plain-text disclaimer must match verbatim.
    text_result = runner.invoke(main, ["models"])
    assert data["note"] in text_result.output


# ---------------------------------------------------------------------------
# Regression: non-JSON output is unchanged
# ---------------------------------------------------------------------------


def test_timeline_text_output_unchanged_when_no_json_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["timeline"])
    assert result.exit_code == 0
    assert "No checkpoints yet." in result.output


def test_memory_list_text_output_unchanged_when_no_json_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["memory", "list"])
    assert result.exit_code == 0
    assert "No memories stored yet." in result.output


def test_runs_text_output_unchanged_when_no_json_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["runs"])
    assert result.exit_code == 0
    assert "No runs recorded yet." in result.output


def test_log_text_output_unchanged_when_no_json_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["log"])
    assert result.exit_code == 0
    assert "No actions recorded yet." in result.output


def test_models_text_output_unchanged_when_no_json_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["models"])
    assert result.exit_code == 0
    assert "MODEL" in result.output
    assert "STREAM" in result.output


# ---------------------------------------------------------------------------
# SCHEMA_VERSION single-source-of-truth
# ---------------------------------------------------------------------------


def test_schema_version_only_defined_in_output_module():
    repo_root = Path(__file__).resolve().parents[1]
    hits = []
    for py_file in (repo_root / "mazu").rglob("*.py"):
        if py_file.name == "output.py":
            continue
        if "schema_version" in py_file.read_text(encoding="utf-8"):
            hits.append(str(py_file))
    assert hits == [], f"schema_version referenced outside mazu/output.py: {hits}"
