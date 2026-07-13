"""Tests for mazu/cli.py's new Faza 4 additions: `mazu --version` and
`mazu checkpoint list`. Uses Click's CliRunner against an isolated filesystem, so
these exercise the real command wiring, not just the underlying functions.
"""

import subprocess
from importlib.metadata import version as installed_version

import pytest
from click.testing import CliRunner

import mazu
from mazu.cli import _memory_db_path, _usage_db_path, main
from mazu.memory.store import MemoryStore
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
