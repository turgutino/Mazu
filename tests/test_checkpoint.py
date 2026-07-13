import subprocess
from pathlib import Path

import pytest

from mazu.checkpoint.manager import CheckpointManager


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    # CI runners (and some dev machines) have no global git user configured;
    # `git commit` fails without one. Scope it to this test's HOME so it never
    # touches the real global git config.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    # Mirrors cli.py's _ensure_gitignore(), which every real command calls before
    # constructing a CheckpointManager. Without it, `.mazu/` (checkpoints + skills)
    # would be tracked/cleaned by git like any other file, which the restore logic
    # doesn't expect -- this fixture keeps the test faithful to actual usage.
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    return tmp_path


def test_snapshot_creates_checkpoint_and_commit(project: Path):
    (project / "a.py").write_text("print('a')")
    manager = CheckpointManager(project)

    entry = manager.snapshot(messages=[{"role": "user", "content": "hi"}], trigger="manual")

    assert entry["id"] == "cp_000001"
    assert (manager.checkpoints_dir / entry["id"] / "conversation.json").exists()


def test_rollback_restores_deleted_file(project: Path):
    (project / "a.py").write_text("print('a')")
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    (project / "a.py").unlink()
    (project / "b.py").write_text("print('b')")

    manager.restore(entry["id"])

    assert (project / "a.py").exists()
    assert not (project / "b.py").exists()


def test_rollback_restores_memory_db(project: Path):
    from mazu.memory.store import MemoryStore

    mazu_dir = project / ".mazu"
    mazu_dir.mkdir()
    store = MemoryStore(mazu_dir / "memory.db")
    store.add(category="fact", title="Before checkpoint", body="x")
    store.close()

    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    store = MemoryStore(mazu_dir / "memory.db")
    store.add(category="fact", title="After checkpoint", body="y")
    store.close()

    manager.restore(entry["id"])

    store = MemoryStore(mazu_dir / "memory.db")
    titles = {row["title"] for row in store.all_active()}
    store.close()
    assert titles == {"Before checkpoint"}


def test_rollback_restores_skills(project: Path):
    from mazu.skills.manager import SkillManager

    skills = SkillManager(project)
    skills.save("skill_one", "does a thing", "def run(args):\n    return 'ok'")

    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    skills.save("skill_two", "does another thing", "def run(args):\n    return 'ok2'")
    assert len(skills.list()) == 2

    manager.restore(entry["id"])

    assert [m["name"] for m in skills.list()] == ["skill_one"]


def test_checkpoint_ids_survive_pruning_without_collision(project: Path):
    manager = CheckpointManager(project, retention=3)
    entries = [manager.snapshot(messages=[], trigger="manual") for _ in range(5)]

    # Pruning already ran inline (snapshot calls prune()); only the last 3 remain on disk.
    remaining = manager.list_checkpoints()
    assert len(remaining) == 3
    assert [e["id"] for e in remaining] == [e["id"] for e in entries[-3:]]

    # A new checkpoint must not reuse an id that collides with history.
    new_entry = manager.snapshot(messages=[], trigger="manual")
    assert new_entry["id"] == "cp_000006"


def test_prune_keeps_only_most_recent(project: Path):
    manager = CheckpointManager(project, retention=100)
    for _ in range(5):
        manager.snapshot(messages=[], trigger="manual")

    pruned_count = manager.prune(keep_last=2)

    assert pruned_count == 3
    assert len(manager.list_checkpoints()) == 2


def test_restore_unknown_checkpoint_raises(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    with pytest.raises(ValueError):
        manager.restore("cp_999999")


def test_is_dirty_detects_uncommitted_changes(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    assert manager.is_dirty() is False

    (project / "new_file.py").write_text("x")
    assert manager.is_dirty() is True


# ---------------------------------------------------------------------------
# timeline / show / diff (Checkpoint UX)
# ---------------------------------------------------------------------------


def test_has_memory_snapshot_true_when_present(project: Path):
    from mazu.memory.store import MemoryStore

    mazu_dir = project / ".mazu"
    mazu_dir.mkdir()
    MemoryStore(mazu_dir / "memory.db").close()

    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    assert manager.has_memory_snapshot(entry["id"]) is True


def test_has_memory_snapshot_false_when_absent(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")
    assert manager.has_memory_snapshot(entry["id"]) is False


def test_has_skills_snapshot_true_when_present(project: Path):
    from mazu.skills.manager import SkillManager

    SkillManager(project).save("s1", "does a thing", "def run(args):\n    return 'ok'")
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    assert manager.has_skills_snapshot(entry["id"]) is True


def test_show_entry_reports_message_count(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(
        messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        trigger="manual",
    )

    shown = manager.show_entry(entry["id"])
    assert shown["message_count"] == 2
    assert shown["id"] == entry["id"]
    assert shown["has_memory_snapshot"] is False


def test_show_entry_defaults_to_most_recent(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")
    second = manager.snapshot(messages=[{"role": "user", "content": "x"}], trigger="manual")

    shown = manager.show_entry()  # no id -- should resolve to the last one
    assert shown["id"] == second["id"]


def test_show_entry_unknown_id_raises(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    with pytest.raises(ValueError):
        manager.show_entry("cp_999999")


def test_diff_against_current_reflects_tracked_file_changes(project: Path):
    (project / "a.py").write_text("print('a')")
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    (project / "a.py").write_text("print('a')\nprint('more')")

    _, diff = manager.diff_against_current(entry["id"])
    assert "a.py" in diff


def test_diff_against_current_includes_untracked_new_files(project: Path):
    """git diff alone never shows untracked files at all -- a file the agent
    created since the checkpoint but never `git add`ed would otherwise silently
    vanish from the diff, which defeats the point of this command.
    """
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    (project / "b.py").write_text("print('b')")  # deliberately never git add'ed

    _, diff = manager.diff_against_current(entry["id"])
    assert "b.py" in diff
    assert "untracked" in diff.lower()


def test_timeline_entries_first_checkpoint_has_no_files_changed(project: Path):
    (project / "a.py").write_text("print('a')")
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")

    timeline = manager.timeline_entries()
    assert len(timeline) == 1
    assert timeline[0]["files_changed"] == []


def test_timeline_entries_shows_changes_since_previous_checkpoint(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")  # cp_000001, nothing yet

    (project / "new_file.py").write_text("x")
    manager.snapshot(messages=[], trigger="manual")  # cp_000002, new_file.py added

    timeline = manager.timeline_entries()
    assert timeline[0]["files_changed"] == []
    assert "new_file.py" in timeline[1]["files_changed"]


def test_timeline_entries_includes_snapshot_flags(project: Path):
    from mazu.memory.store import MemoryStore

    mazu_dir = project / ".mazu"
    mazu_dir.mkdir()
    MemoryStore(mazu_dir / "memory.db").close()

    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")

    timeline = manager.timeline_entries()
    assert timeline[0]["has_memory_snapshot"] is True
    assert timeline[0]["has_skills_snapshot"] is False


def test_timeline_entries_empty_when_no_checkpoints(project: Path):
    manager = CheckpointManager(project)
    assert manager.timeline_entries() == []


def test_preview_rollback_shows_uncommitted_changes_to_the_latest_checkpoint(project: Path):
    """Regression test: `git diff <commit> HEAD` (two explicit commit refs) shows
    nothing when `commit` IS HEAD, even with real uncommitted working-tree changes
    -- since both sides of the diff are the same commit. preview_rollback (used by
    `mazu rollback`) must still show what would actually be discarded.
    """
    (project / "a.py").write_text("print('a')")
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")  # this becomes HEAD

    (project / "a.py").write_text("print('a')\nprint('uncommitted edit')")

    _, diff = manager.preview_rollback(entry["id"])
    assert "a.py" in diff


# ---------------------------------------------------------------------------
# inspect / compare / branch-from (rest of Phase B)
# ---------------------------------------------------------------------------


def test_inspect_memory_reads_from_the_snapshot_not_live_db(project: Path):
    from mazu.memory.store import MemoryStore

    mazu_dir = project / ".mazu"
    mazu_dir.mkdir()
    store = MemoryStore(mazu_dir / "memory.db")
    store.add(category="fact", title="Before checkpoint", body="x")
    store.close()

    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    # Add a fact AFTER the checkpoint -- inspect_memory must NOT see this, since
    # it reads the frozen snapshot, not the live (now-ahead) memory.db.
    store = MemoryStore(mazu_dir / "memory.db")
    store.add(category="fact", title="After checkpoint", body="y")
    store.close()

    memories = manager.inspect_memory(entry["id"])
    titles = {m["title"] for m in memories}
    assert titles == {"Before checkpoint"}


def test_inspect_memory_empty_when_no_snapshot(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")
    assert manager.inspect_memory(entry["id"]) == []


def test_inspect_conversation_returns_the_captured_messages(project: Path):
    manager = CheckpointManager(project)
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    entry = manager.snapshot(messages=messages, trigger="manual")

    assert manager.inspect_conversation(entry["id"]) == messages


def test_inspect_conversation_empty_when_no_messages(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")
    assert manager.inspect_conversation(entry["id"]) == []


def test_compare_shows_diff_between_two_checkpoints(project: Path):
    manager = CheckpointManager(project)
    entry_a = manager.snapshot(messages=[], trigger="manual")

    (project / "new_file.py").write_text("x")
    entry_b = manager.snapshot(messages=[], trigger="manual")

    a, b, diff = manager.compare(entry_a["id"], entry_b["id"])
    assert a["id"] == entry_a["id"]
    assert b["id"] == entry_b["id"]
    assert "new_file.py" in diff


def test_compare_identical_checkpoints_shows_no_diff(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    _, _, diff = manager.compare(entry["id"], entry["id"])
    assert diff.strip() == ""


def test_compare_unknown_id_raises(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")
    with pytest.raises(ValueError):
        manager.compare(entry["id"], "cp_999999")


def test_branch_from_creates_branch_at_checkpoint_commit(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    manager.branch_from(entry["id"], "my-experiment")

    result = subprocess.run(
        ["git", "branch", "--list", "my-experiment"], cwd=project, capture_output=True, text=True
    )
    assert "my-experiment" in result.stdout


def test_branch_from_does_not_touch_current_branch_or_working_tree(project: Path):
    (project / "a.py").write_text("original")
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")

    (project / "a.py").write_text("modified after checkpoint")  # uncommitted change

    manager.branch_from(entry["id"], "my-experiment")

    # Still on main/master, uncommitted change untouched -- branch_from must not
    # check out the new branch or reset anything, unlike restore().
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=project, capture_output=True, text=True
    ).stdout.strip()
    assert current_branch != "my-experiment"
    assert (project / "a.py").read_text() == "modified after checkpoint"


def test_branch_from_duplicate_name_raises_clean_error(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")
    manager.branch_from(entry["id"], "my-experiment")

    with pytest.raises(ValueError):
        manager.branch_from(entry["id"], "my-experiment")


def test_branch_from_unknown_checkpoint_raises(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    with pytest.raises(ValueError):
        manager.branch_from("cp_999999", "my-experiment")


# ---------------------------------------------------------------------------
# session_id tracking + latest_for_session (Phase F: run resume)
# ---------------------------------------------------------------------------


def test_snapshot_stores_session_id_in_the_entry(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual", session_id="run-1")
    assert entry["session_id"] == "run-1"


def test_snapshot_without_session_id_stores_none(project: Path):
    manager = CheckpointManager(project)
    entry = manager.snapshot(messages=[], trigger="manual")
    assert entry["session_id"] is None


def test_latest_for_session_returns_the_highest_step_match(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[{"n": 1}], trigger="manual", session_id="run-1")
    second = manager.snapshot(messages=[{"n": 2}], trigger="manual", session_id="run-1")
    manager.snapshot(messages=[{"n": 3}], trigger="manual", session_id="run-2")

    latest = manager.latest_for_session("run-1")
    assert latest["id"] == second["id"]


def test_latest_for_session_ignores_other_sessions(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual", session_id="run-1")

    assert manager.latest_for_session("run-2") is None


def test_latest_for_session_ignores_checkpoints_with_no_session_id(project: Path):
    manager = CheckpointManager(project)
    manager.snapshot(messages=[], trigger="manual")  # no session_id, e.g. a manual chat checkpoint

    assert manager.latest_for_session("run-1") is None


def test_latest_for_session_no_checkpoints_at_all_returns_none(project: Path):
    manager = CheckpointManager(project)
    manager.ensure_git_repo()
    assert manager.latest_for_session("run-1") is None
