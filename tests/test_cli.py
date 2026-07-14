"""Tests for mazu/cli.py's new Faza 4 additions: `mazu --version` and
`mazu checkpoint list`. Uses Click's CliRunner against an isolated filesystem, so
these exercise the real command wiring, not just the underlying functions.
"""

import subprocess
from importlib.metadata import version as installed_version

import pytest
from click.testing import CliRunner

import mazu
import mazu.cli as cli_module
from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.cli import (
    _action_log_db_path,
    _memory_db_path,
    _parse_shell_allowlist,
    _runs_db_path,
    _usage_db_path,
    main,
)
from mazu.memory.store import MemoryStore
from mazu.runs.store import RunStore
from mazu.usage.store import UsageStore


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def test_version_flag_reports_the_real_installed_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "mazu" in result.output.lower()
    assert installed_version("mazu") in result.output


def test_mazu_dunder_version_matches_installed_package():
    # Regression test for a real bug: mazu/__init__.py used to hardcode
    # __version__ = "0.1.0" as a plain string that nothing ever kept in sync with
    # pyproject.toml's actual version, so it silently drifted out of date on every
    # release after the first. It must now be derived from package metadata.
    assert mazu.__version__ == installed_version("mazu")


def test_checkpoint_list_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["checkpoint", "list"])

    assert result.exit_code == 0
    assert "No checkpoints yet." in result.output


def test_checkpoint_list_shows_created_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["checkpoint"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(main, ["checkpoint", "list"])
    assert result.exit_code == 0, result.output
    assert "cp_000001" in result.output


def test_usage_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["usage"])

    assert result.exit_code == 0
    assert "No usage recorded yet." in result.output


def test_usage_shows_logged_spend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # _usage_db_path() resolves via Path.home(), which respects HOME/USERPROFILE --
    # both already redirected to tmp_path by the autouse _git_identity fixture.
    store = UsageStore(_usage_db_path())
    store.log("chat", "s1", "anthropic", "claude-sonnet-5", 1000, 500, 0.0105)
    store.log("run", "s2", "deepseek", "deepseek-chat", 2000, 1000, 0.0016)
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["usage"])

    assert result.exit_code == 0, result.output
    assert "claude-sonnet-5" in result.output
    assert "deepseek-chat" in result.output
    assert "0.0121" in result.output  # total = 0.0105 + 0.0016


def test_doctor_reports_problems(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])

    assert result.exit_code == 0
    assert "[FAIL]" in result.output
    assert "problem(s) found" in result.output


def test_doctor_all_good_with_key_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])

    assert result.exit_code == 0
    assert "[FAIL]" not in result.output


def test_doctor_live_flag_does_not_crash_without_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--live"])

    assert result.exit_code == 0
    assert "(live)" not in result.output  # nothing to live-check with no keys set


def test_doctor_fix_creates_gitignore_and_git_repo(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--fix"])

    assert result.exit_code == 0, result.output
    assert "[fix]" in result.output
    assert (tmp_path / ".git").exists()
    assert ".mazu/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "[FAIL]" not in result.output


def test_doctor_fix_reports_nothing_to_fix_when_already_correct(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--fix"])

    assert result.exit_code == 0, result.output
    assert "Nothing to fix." in result.output


def test_usage_since_days_filters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = UsageStore(_usage_db_path())
    store.conn.execute(
        "INSERT INTO usage_log "
        "(created_at, command, session_id, provider, model, input_tokens, output_tokens, estimated_cost_usd) "
        "VALUES ('2020-01-01T00:00:00+00:00', 'chat', 's-old', 'anthropic', 'claude-sonnet-5', 100, 50, 1.0)"
    )
    store.conn.commit()
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["usage", "--since-days", "7"])

    assert result.exit_code == 0, result.output
    assert "No usage recorded yet." in result.output  # the only row is far outside the window


def test_memory_consolidate_no_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "consolidate"])

    assert result.exit_code == 0
    assert "No near-duplicate memories found." in result.output


def test_memory_consolidate_dry_run_does_not_modify(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency and JSON")
    store.add(
        category="decision", title="PostgreSQL for storage", body="For concurrency and JSON"
    )
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "consolidate", "--dry-run"])

    assert result.exit_code == 0
    assert "Would merge 1 group" in result.output
    assert "dry run" in result.output

    store = MemoryStore(_memory_db_path(tmp_path))
    assert len(store.all_active()) == 2  # nothing actually changed
    store.close()


def test_memory_consolidate_applies_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    store.add(category="decision", title="Use PostgreSQL", body="For concurrency and JSON")
    store.add(
        category="decision", title="PostgreSQL for storage", body="For concurrency and JSON"
    )
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "consolidate"])

    assert result.exit_code == 0
    assert "Merged 1 group" in result.output

    store = MemoryStore(_memory_db_path(tmp_path))
    assert len(store.all_active()) == 1
    store.close()


def test_memory_consolidate_global_flag_uses_global_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # HOME already redirected here by _git_identity fixture
    from mazu.cli import _global_memory_db_path

    store = MemoryStore(_global_memory_db_path())
    store.add(category="user_preference", title="Name", body="Turgut")
    store.add(category="user_preference", title="User's name", body="Turgut")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "consolidate", "--global"])

    assert result.exit_code == 0
    assert "Merged 1 group" in result.output

    # The project-local store (empty, untouched) proves --global routed correctly.
    project_store = MemoryStore(_memory_db_path(tmp_path))
    assert project_store.all_active() == []
    project_store.close()


# ---------------------------------------------------------------------------
# timeline / checkpoint show / checkpoint diff (Checkpoint UX)
# ---------------------------------------------------------------------------


def test_timeline_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["timeline"])

    assert result.exit_code == 0
    assert "No checkpoints yet." in result.output


def test_timeline_shows_files_changed_between_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    runner.invoke(main, ["checkpoint"])
    (tmp_path / "new_file.py").write_text("x")
    result = runner.invoke(main, ["checkpoint"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(main, ["timeline"])
    assert result.exit_code == 0, result.output
    assert "cp_000001" in result.output
    assert "cp_000002" in result.output
    assert "new_file.py" in result.output


def test_checkpoint_show_unknown_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])  # ensure a git repo + at least one checkpoint exist

    result = runner.invoke(main, ["checkpoint", "show", "cp_999999"])
    assert result.exit_code == 0  # reports the error, doesn't crash
    assert "No checkpoint found" in result.output


def test_checkpoint_show_defaults_to_most_recent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])

    result = runner.invoke(main, ["checkpoint", "show"])
    assert result.exit_code == 0, result.output
    assert "cp_000001" in result.output
    assert "Memory snapshot:" in result.output


def test_checkpoint_diff_shows_untracked_new_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])

    (tmp_path / "brand_new.py").write_text("x")

    result = runner.invoke(main, ["checkpoint", "diff"])
    assert result.exit_code == 0, result.output
    assert "brand_new.py" in result.output
    assert "untracked" in result.output.lower()


def test_checkpoint_diff_no_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])

    result = runner.invoke(main, ["checkpoint", "diff"])
    assert result.exit_code == 0, result.output
    assert "(no changes)" in result.output


# ---------------------------------------------------------------------------
# checkpoint inspect / compare / branch-from (rest of Phase B)
# ---------------------------------------------------------------------------


def test_checkpoint_inspect_requires_a_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])

    result = runner.invoke(main, ["checkpoint", "inspect"])
    assert result.exit_code == 0
    assert "--memory" in result.output


def test_checkpoint_inspect_memory_shows_captured_facts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from mazu.cli import _memory_db_path

    store = MemoryStore(_memory_db_path(tmp_path))
    store.add(category="decision", title="Use PostgreSQL", body="for concurrency")
    store.close()

    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])

    result = runner.invoke(main, ["checkpoint", "inspect", "--memory"])
    assert result.exit_code == 0, result.output
    assert "Use PostgreSQL" in result.output


def test_checkpoint_inspect_conversation_shows_captured_messages(tmp_path, monkeypatch):
    # checkpoint inspect --conversation reads conversation.json, which is only
    # populated with real messages via a live session -- the CLI's bare
    # `mazu checkpoint` snapshot always passes an empty message list. Exercise the
    # manager directly here to prove the CLI wiring formats real messages
    # correctly, same pattern as other tests that seed data below the CLI layer.
    monkeypatch.chdir(tmp_path)
    from mazu.checkpoint.manager import CheckpointManager

    manager = CheckpointManager(tmp_path)
    manager.snapshot(
        messages=[{"role": "user", "content": "hello there"}], trigger="manual"
    )

    runner = CliRunner()
    result = runner.invoke(main, ["checkpoint", "inspect", "--conversation"])
    assert result.exit_code == 0, result.output
    assert "hello there" in result.output


def test_checkpoint_compare_shows_diff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])
    (tmp_path / "new_file.py").write_text("x")
    runner.invoke(main, ["checkpoint"])

    result = runner.invoke(main, ["checkpoint", "compare", "cp_000001", "cp_000002"])
    assert result.exit_code == 0, result.output
    assert "new_file.py" in result.output


def test_checkpoint_compare_unknown_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])

    result = runner.invoke(main, ["checkpoint", "compare", "cp_000001", "cp_999999"])
    assert result.exit_code == 0
    assert "No checkpoint found" in result.output


def test_branch_from_creates_branch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])

    result = runner.invoke(main, ["branch-from", "cp_000001", "my-experiment"])
    assert result.exit_code == 0, result.output
    assert "my-experiment" in result.output

    branches = subprocess.run(
        ["git", "branch", "--list", "my-experiment"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "my-experiment" in branches.stdout


def test_branch_from_unknown_checkpoint_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["checkpoint"])

    result = runner.invoke(main, ["branch-from", "cp_999999", "my-experiment"])
    assert result.exit_code == 0
    assert "No checkpoint found" in result.output


# ---------------------------------------------------------------------------
# Memory UX (Phase C): why / pin / unpin / edit / supersede / stats
# ---------------------------------------------------------------------------


def test_memory_why_shows_included_and_excluded_memories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    store.add(category="decision", title="Use PostgreSQL", body="for the database")
    store.add(category="decision", title="Adopted React", body="for frontend components")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "why", "what database do we use"])

    assert result.exit_code == 0, result.output
    assert "Use PostgreSQL" in result.output
    assert "Adopted React" in result.output
    assert "[x]" in result.output  # at least one included row


def test_memory_why_empty_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["memory", "why", "anything"])

    assert result.exit_code == 0
    assert "No memories stored yet." in result.output


def test_memory_why_marks_pinned_memory_with_its_reason(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    store.add(category="fact", title="Pinned fact", body="x", pinned=True)
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "why", "unrelated query"])

    assert result.exit_code == 0, result.output
    assert "pinned" in result.output.lower()


def test_memory_pin_and_unpin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    memory_id = store.add(category="fact", title="A", body="a")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "pin", str(memory_id)])
    assert result.exit_code == 0, result.output
    assert f"Pinned memory {memory_id}" in result.output

    store = MemoryStore(_memory_db_path(tmp_path))
    assert store.get(memory_id)["pinned"] == 1
    store.close()

    result = runner.invoke(main, ["memory", "unpin", str(memory_id)])
    assert result.exit_code == 0, result.output
    assert f"Unpinned memory {memory_id}" in result.output

    store = MemoryStore(_memory_db_path(tmp_path))
    assert store.get(memory_id)["pinned"] == 0
    store.close()


def test_memory_pin_missing_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["memory", "pin", "9999"])

    assert result.exit_code == 0
    assert "No memory with id 9999" in result.output


def test_memory_edit_updates_title_and_body(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    memory_id = store.add(category="fact", title="Old", body="Old body")
    store.close()

    runner = CliRunner()
    result = runner.invoke(
        main, ["memory", "edit", str(memory_id), "--title", "New", "--body", "New body"]
    )
    assert result.exit_code == 0, result.output
    assert f"Updated memory {memory_id}" in result.output

    store = MemoryStore(_memory_db_path(tmp_path))
    row = store.get(memory_id)
    assert row["title"] == "New"
    assert row["body"] == "New body"
    store.close()


def test_memory_edit_without_flags_is_a_usage_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["memory", "edit", "1"])

    assert result.exit_code != 0
    assert "Provide --title and/or --body" in result.output


def test_memory_edit_missing_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["memory", "edit", "9999", "--title", "X"])

    assert result.exit_code == 0
    assert "No memory with id 9999" in result.output


def test_memory_supersede_retires_old_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    old_id = store.add(category="decision", title="Use MySQL", body="Initial choice")
    new_id = store.add(category="decision", title="Use PostgreSQL", body="Better fit")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "supersede", str(old_id), str(new_id)])
    assert result.exit_code == 0, result.output
    assert f"Memory {old_id} marked as superseded by {new_id}" in result.output

    store = MemoryStore(_memory_db_path(tmp_path))
    active_ids = {row["id"] for row in store.all_active()}
    assert old_id not in active_ids
    assert new_id in active_ids
    store.close()


def test_memory_supersede_unknown_old_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    new_id = store.add(category="decision", title="Use PostgreSQL", body="Better fit")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "supersede", "9999", str(new_id)])
    assert result.exit_code == 0
    assert "No memory with id 9999" in result.output


def test_memory_supersede_unknown_new_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    old_id = store.add(category="decision", title="Use MySQL", body="Initial choice")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "supersede", str(old_id), "9999"])
    assert result.exit_code == 0
    assert "No memory with id 9999" in result.output


def test_memory_stats_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["memory", "stats"])

    assert result.exit_code == 0, result.output
    assert "Total: 0" in result.output


def test_memory_stats_counts_by_category_and_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(_memory_db_path(tmp_path))
    store.add(category="decision", title="A", body="a", source="explicit")
    store.add(category="decision", title="B", body="b", source="auto_extracted")
    store.add(category="mistake", title="C", body="c", source="explicit", pinned=True)
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "stats"])

    assert result.exit_code == 0, result.output
    assert "Total: 3" in result.output
    assert "1 pinned" in result.output
    assert "decision: 2" in result.output
    assert "mistake: 1" in result.output
    assert "explicit: 2" in result.output
    assert "auto_extracted: 1" in result.output


def test_memory_stats_global_flag_uses_global_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # HOME already redirected here by _git_identity fixture
    from mazu.cli import _global_memory_db_path

    store = MemoryStore(_global_memory_db_path())
    store.add(category="user_preference", title="Name", body="Turgut")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "stats", "--global"])

    assert result.exit_code == 0, result.output
    assert "Total: 1" in result.output


def test_memory_list_shows_retrieval_usage_after_context_build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    from mazu.memory.retrieval import build_context_block

    store = MemoryStore(_memory_db_path(tmp_path))
    store.add(category="decision", title="Use PostgreSQL", body="for the database")
    build_context_block(store, query="what database do we use")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory", "list"])

    assert result.exit_code == 0, result.output
    assert "used 1x" in result.output


# ---------------------------------------------------------------------------
# Agent Action Log (Phase D): mazu log / mazu log show
# ---------------------------------------------------------------------------


def test_log_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["log"])

    assert result.exit_code == 0
    assert "No actions recorded yet." in result.output


def test_log_lists_recent_sessions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ActionLogStore(_action_log_db_path(tmp_path))
    store.log("s1", "chat", "read_file", '{"path": "a.py"}', "ok", "contents", None)
    store.log("s1", "chat", "write_file", '{"path": "a.py"}', "error", "boom", "a.py")
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["log"])

    assert result.exit_code == 0, result.output
    assert "s1" in result.output
    assert "(chat)" in result.output
    assert "2 action(s)" in result.output
    assert "1 not-ok" in result.output


def test_log_show_unknown_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["log", "show", "nope"])

    assert result.exit_code == 0
    assert "No actions recorded for session nope." in result.output


def test_log_show_displays_full_action_detail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ActionLogStore(_action_log_db_path(tmp_path))
    store.log(
        "s1", "run", "write_file", '{"path": "a.py"}', "ok", "Wrote 5 bytes to a.py", "a.py"
    )
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["log", "show", "s1"])

    assert result.exit_code == 0, result.output
    assert "write_file" in result.output
    assert "ok" in result.output
    assert "Wrote 5 bytes to a.py" in result.output
    assert "a.py" in result.output


def test_log_show_only_shows_the_requested_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ActionLogStore(_action_log_db_path(tmp_path))
    store.log("s1", "chat", "read_file", "{}", "ok", "s1 output", None)
    store.log("s2", "chat", "read_file", "{}", "ok", "s2 output", None)
    store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["log", "show", "s1"])

    assert result.exit_code == 0, result.output
    assert "s1 output" in result.output
    assert "s2 output" not in result.output


# ---------------------------------------------------------------------------
# Safer Execution (Phase E): --dry-run, --shell-allowlist
# ---------------------------------------------------------------------------


def test_parse_shell_allowlist_none_when_unset():
    assert _parse_shell_allowlist(None) is None
    assert _parse_shell_allowlist("") is None


def test_parse_shell_allowlist_splits_and_trims():
    assert _parse_shell_allowlist("git, npm , pytest") == ["git", "npm", "pytest"]


def test_parse_shell_allowlist_drops_empty_entries():
    assert _parse_shell_allowlist("git,,npm,") == ["git", "npm"]


def test_run_dry_run_flag_reaches_run_autonomous(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    captured = {}

    def _fake_run_autonomous(registry, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "run_autonomous", _fake_run_autonomous)

    runner = CliRunner()
    result = runner.invoke(main, ["run", "do something", "--dry-run", "--model", "deepseek:deepseek-chat"])

    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True


def test_run_without_dry_run_flag_defaults_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    captured = {}

    def _fake_run_autonomous(registry, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "run_autonomous", _fake_run_autonomous)

    runner = CliRunner()
    result = runner.invoke(main, ["run", "do something", "--model", "deepseek:deepseek-chat"])

    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is False


def test_run_shell_allowlist_flag_reaches_run_autonomous(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    captured = {}

    def _fake_run_autonomous(registry, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "run_autonomous", _fake_run_autonomous)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "do something", "--shell-allowlist", "git,npm", "--model", "deepseek:deepseek-chat"],
    )

    assert result.exit_code == 0, result.output
    assert captured["shell_allowlist"] == ["git", "npm"]


def test_chat_shell_allowlist_flag_reaches_run_chat_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    captured = {}

    def _fake_run_chat_loop(registry, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "run_chat_loop", _fake_run_chat_loop)

    runner = CliRunner()
    result = runner.invoke(
        main, ["chat", "--shell-allowlist", "git,npm", "--model", "deepseek:deepseek-chat"]
    )

    assert result.exit_code == 0, result.output
    assert captured["shell_allowlist"] == ["git", "npm"]


def test_run_help_documents_dry_run_and_shell_allowlist():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])

    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--shell-allowlist" in result.output


# ---------------------------------------------------------------------------
# Better Autonomous Runs (Phase F): mazu run --resume, mazu runs
# ---------------------------------------------------------------------------


def test_run_without_task_or_resume_is_a_usage_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run"])

    assert result.exit_code != 0
    assert "Provide a TASK" in result.output


def test_run_with_both_task_and_resume_is_a_usage_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "do something", "--resume", "r1"])

    assert result.exit_code != 0
    assert "not both" in result.output


def test_run_resume_unknown_run_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--resume", "nope"])

    assert result.exit_code == 0
    assert "No run found with id nope." in result.output


def test_run_resume_with_no_checkpoint_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_store = RunStore(_runs_db_path(tmp_path))
    run_store.start("r1", "do something", "deepseek:deepseek-chat", 15, 1, False, None, None, False)
    run_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--resume", "r1"])

    assert result.exit_code == 0
    assert "No checkpoint found for run r1" in result.output


def test_run_resume_recovers_stored_config_and_reaches_run_autonomous(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    run_store = RunStore(_runs_db_path(tmp_path))
    run_store.start(
        "r1", "original task", "deepseek:deepseek-chat", 8, 2, True, ["git", "npm"], 1.5, False
    )
    run_store.close()

    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_manager.snapshot(
        messages=[{"role": "user", "content": "original task"}],
        trigger="auto_after_tool_round",
        session_id="r1",
    )

    captured = {}

    def _fake_run_autonomous(registry, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "run_autonomous", _fake_run_autonomous)

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--resume", "r1"])

    assert result.exit_code == 0, result.output
    assert captured["task"] == "original task"
    assert captured["model"] == "deepseek:deepseek-chat"
    assert captured["max_steps"] == 8
    assert captured["checkpoint_every"] == 2
    assert captured["allow_shell"] is True
    assert captured["shell_allowlist"] == ["git", "npm"]
    assert captured["max_cost"] == 1.5
    assert captured["dry_run"] is False
    assert captured["session_id"] == "r1"
    assert captured["resume_messages"] == [{"role": "user", "content": "original task"}]
    assert "Resuming run r1" in result.output


def test_runs_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["runs"])

    assert result.exit_code == 0
    assert "No runs recorded yet." in result.output


def test_runs_lists_recorded_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_store = RunStore(_runs_db_path(tmp_path))
    run_store.start("r1", "do something", "deepseek:deepseek-chat", 15, 1, False, None, None, False)
    run_store.finish("r1", status="completed", stop_reason="end_turn", memories_saved=1)
    run_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["runs"])

    assert result.exit_code == 0, result.output
    assert "r1" in result.output
    assert "completed" in result.output
    assert "end_turn" in result.output


# ---------------------------------------------------------------------------
# Provider Layer (Phase G): mazu models, mazu config
# ---------------------------------------------------------------------------


def test_models_lists_every_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["models"])

    assert result.exit_code == 0, result.output
    assert "anthropic:claude-sonnet-5" in result.output
    assert "openai:gpt-5" in result.output
    assert "deepseek:deepseek-chat" in result.output
    assert "gemini:gemini-2.0-flash" in result.output


def test_config_list_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "list"])

    assert result.exit_code == 0
    assert "No config set." in result.output


def test_config_set_and_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["config", "set", "default_model", "deepseek:deepseek-chat"])
    assert result.exit_code == 0, result.output
    assert "Set default_model = deepseek:deepseek-chat" in result.output

    result = runner.invoke(main, ["config", "list"])
    assert "default_model = deepseek:deepseek-chat" in result.output


def test_config_set_api_key_is_masked_in_output_and_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["config", "set", "anthropic_api_key", "sk-ant-1234567890abcdef"])
    assert result.exit_code == 0, result.output
    assert "abcdef" not in result.output  # only the last 4 chars should ever show
    assert "cdef" in result.output

    result = runner.invoke(main, ["config", "list"])
    assert "sk-ant-1234567890abcdef" not in result.output
    assert "cdef" in result.output


def test_config_set_unknown_key_is_a_usage_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "set", "bogus_key", "x"])

    assert result.exit_code != 0
    assert "Unknown config key" in result.output


def test_config_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["config", "set", "default_model", "deepseek:deepseek-chat"])

    result = runner.invoke(main, ["config", "unset", "default_model"])
    assert result.exit_code == 0
    assert "Unset default_model." in result.output

    result = runner.invoke(main, ["config", "list"])
    assert "No config set." in result.output


def test_config_unset_missing_key_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "unset", "default_model"])

    assert result.exit_code == 0
    assert "was not set" in result.output


def test_config_set_default_model_affects_chat_without_explicit_model_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAZU_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    runner = CliRunner()
    runner.invoke(main, ["config", "set", "default_model", "deepseek:deepseek-chat"])

    captured = {}

    def _fake_run_chat_loop(registry, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "run_chat_loop", _fake_run_chat_loop)
    result = runner.invoke(main, ["chat"])

    assert result.exit_code == 0, result.output
    assert captured["model"] is None  # --model wasn't passed -- resolution happens downstream
    # The real proof this worked: ensure_api_key(None) didn't raise SystemExit, which
    # it would have if default_model() hadn't picked up deepseek from config and
    # instead fallen through to the hardcoded Anthropic default with no key set.


# ---------------------------------------------------------------------------
# Install & Onboarding (Phase H): mazu setup
# ---------------------------------------------------------------------------


def _git_ready(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


# Config-value assertions below go through mazu.config.list_config() rather than
# reading a guessed file path directly -- config_path() is redirected by the global
# autouse fixture in tests/conftest.py (an isolated tmp_path, not based on HOME), so
# list_config() is the only way to see exactly what a test actually wrote, matching
# how the production code itself reads it back.


def test_setup_saves_key_and_declines_verify_and_init(tmp_path, monkeypatch):
    from mazu.config import list_config

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    runner = CliRunner()
    # provider=deepseek, key=..., verify=n, default_model=n, init=n
    result = runner.invoke(main, ["setup"], input="deepseek\nsk-test-123\nn\nn\nn\n")

    assert result.exit_code == 0, result.output
    assert "Saved to" in result.output
    assert list_config()["deepseek_api_key"] == "sk-test-123"
    assert not (project / ".mazu").exists()  # declined init


def test_setup_sets_default_model_when_confirmed(tmp_path, monkeypatch):
    from mazu.config import list_config

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(main, ["setup"], input="anthropic\nsk-test-123\nn\ny\nn\n")

    assert result.exit_code == 0, result.output
    assert "default_model set to anthropic:claude-sonnet-5" in result.output
    assert list_config()["default_model"] == "anthropic:claude-sonnet-5"


def test_setup_declines_default_model(tmp_path, monkeypatch):
    from mazu.config import list_config

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(main, ["setup"], input="anthropic\nsk-test-123\nn\nn\nn\n")

    assert result.exit_code == 0, result.output
    assert "default_model" not in list_config()


def test_setup_initializes_project_when_confirmed(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    _git_ready(project, monkeypatch)

    runner = CliRunner()
    result = runner.invoke(main, ["setup"], input="deepseek\nsk-test-123\nn\nn\ny\n")

    assert result.exit_code == 0, result.output
    assert (project / ".mazu").exists()
    assert "Initialized Mazu project memory" in result.output


def test_setup_skips_init_prompt_when_already_initialized(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".mazu").mkdir()
    monkeypatch.chdir(project)

    runner = CliRunner()
    # Only 4 answers needed (provider, key, verify, default_model) -- init must NOT
    # be prompted; if it were, this input sequence would be misapplied to the wrong
    # prompt and the command would hang/fail on missing input.
    result = runner.invoke(main, ["setup"], input="deepseek\nsk-test-123\nn\nn\n")

    assert result.exit_code == 0, result.output
    assert "Initialize Mazu in the current directory" not in result.output


def test_setup_live_verify_success_path(tmp_path, monkeypatch):
    from mazu.diagnostics import CheckResult

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    monkeypatch.setattr(
        cli_module,
        "check_live_api_key",
        lambda provider_name, model: CheckResult(f"{provider_name} (live)", "ok", "authenticated successfully"),
    )

    runner = CliRunner()
    result = runner.invoke(main, ["setup"], input="deepseek\nsk-test-123\ny\nn\nn\n")

    assert result.exit_code == 0, result.output
    assert "[OK] authenticated successfully" in result.output


def test_setup_live_verify_failure_path_still_keeps_the_key(tmp_path, monkeypatch):
    from mazu.config import list_config
    from mazu.diagnostics import CheckResult

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    monkeypatch.setattr(
        cli_module,
        "check_live_api_key",
        lambda provider_name, model: CheckResult(f"{provider_name} (live)", "fail", "key rejected: bad key"),
    )

    runner = CliRunner()
    result = runner.invoke(main, ["setup"], input="deepseek\nsk-test-123\ny\nn\nn\n")

    assert result.exit_code == 0, result.output
    assert "[FAIL] key rejected" in result.output
    assert "still saved" in result.output
    assert list_config()["deepseek_api_key"] == "sk-test-123"


def test_setup_help_documents_the_command():
    runner = CliRunner()
    result = runner.invoke(main, ["setup", "--help"])
    assert result.exit_code == 0
    assert "wizard" in result.output.lower()
