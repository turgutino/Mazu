from click.testing import CliRunner

from mazu.cli import main
from mazu.config import set_config_value
from mazu.curator.store import CuratorStore, new_curator_run_id
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager


def test_skills_list_archived_flag_shows_only_archived(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = SkillManager(tmp_path)
    manager.save("active_skill", "desc", "def run(args):\n    return 'ok'\n")
    manager.save("archived_skill", "desc", "def run(args):\n    return 'ok'\n")
    manager.archive("archived_skill")

    runner = CliRunner()
    result = runner.invoke(main, ["skills", "list"])
    assert "active_skill" in result.output
    assert "archived_skill" not in result.output

    result = runner.invoke(main, ["skills", "list", "--archived"])
    assert "archived_skill" in result.output
    assert "active_skill" not in result.output


def test_skills_unarchive_restores_visibility(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", "def run(args):\n    return 'ok'\n")
    manager.archive("s1")

    runner = CliRunner()
    result = runner.invoke(main, ["skills", "unarchive", "s1"])
    assert "Unarchived skill 's1'" in result.output

    result = runner.invoke(main, ["skills", "list"])
    assert "s1" in result.output


def test_skills_unarchive_unknown_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["skills", "unarchive", "nope"])
    assert "No skill named 'nope'" in result.output


def test_curator_status_reports_not_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["curator", "status"])
    assert "not configured" in result.output


def test_curator_run_is_a_noop_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["curator", "run"])
    assert "not configured" in result.output
    assert not (tmp_path / ".mazu" / "curator.db").exists()


def test_curator_status_reports_configured_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")

    runner = CliRunner()
    result = runner.invoke(main, ["curator", "status"])
    assert "anthropic:claude-haiku-4-5" in result.output


def test_curator_enable_disable_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(main, ["curator", "disable"])
    result = runner.invoke(main, ["curator", "status"])
    # disable doesn't require curator to be configured first; status still
    # reports "not configured" here since no key/model were ever set.
    assert result.exit_code == 0

    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")
    runner.invoke(main, ["curator", "disable"])
    result = runner.invoke(main, ["curator", "status"])
    assert "Enabled: False" in result.output

    runner.invoke(main, ["curator", "enable"])
    result = runner.invoke(main, ["curator", "status"])
    assert "Enabled: True" in result.output


def test_curator_log_and_report_with_no_runs_yet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["curator", "log"])
    assert "No curator runs yet" in result.output
    result = runner.invoke(main, ["curator", "report"])
    assert "No curator runs yet" in result.output


def test_curator_undo_with_no_runs_yet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["curator", "undo", "1"])
    assert "No curator runs yet" in result.output


def test_curator_undo_unarchives_a_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory_store = MemoryStore(tmp_path / ".mazu" / "memory.db")
    memory_id = memory_store.add(category="fact", title="Old fact", body="body")
    memory_store.archive(memory_id)
    memory_store.close()

    curator_store = CuratorStore(tmp_path / ".mazu" / "curator.db")
    run_id = new_curator_run_id()
    log_id = curator_store.log_entry(
        run_id=run_id, area="memory", action="archive_memory", target_type="memory",
        target_id=str(memory_id), rationale="stale", reversal_hint=f"mazu memory unarchive {memory_id}",
    )
    curator_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["curator", "undo", str(log_id)])
    assert "Reversed" in result.output

    memory_store = MemoryStore(tmp_path / ".mazu" / "memory.db")
    active_ids = {r["id"] for r in memory_store.all_active()}
    memory_store.close()
    assert memory_id in active_ids


def test_curator_undo_unarchives_a_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", "def run(args):\n    return 'ok'\n")
    manager.archive("s1")

    curator_store = CuratorStore(tmp_path / ".mazu" / "curator.db")
    run_id = new_curator_run_id()
    log_id = curator_store.log_entry(
        run_id=run_id, area="skills", action="archive_skill", target_type="skill",
        target_id="s1", rationale="failing", reversal_hint="mazu skills unarchive s1",
    )
    curator_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["curator", "undo", str(log_id)])
    assert "Reversed" in result.output

    manager2 = SkillManager(tmp_path)
    assert "s1" in {m["name"] for m in manager2.list()}


def test_curator_undo_prints_guidance_for_non_reversible_action(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    curator_store = CuratorStore(tmp_path / ".mazu" / "curator.db")
    run_id = new_curator_run_id()
    log_id = curator_store.log_entry(
        run_id=run_id, area="memory", action="edit_memory", target_type="memory",
        target_id="5", rationale="fixed typo", reversal_hint='mazu memory edit 5 --title "Old" --body "Old body"',
    )
    curator_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["curator", "undo", str(log_id)])
    assert "No automatic undo" in result.output
    assert "mazu memory edit 5" in result.output


def test_curator_undo_unknown_log_id_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    curator_store = CuratorStore(tmp_path / ".mazu" / "curator.db")
    curator_store.close()

    runner = CliRunner()
    result = runner.invoke(main, ["curator", "undo", "9999"])
    assert "No curator log entry with id 9999" in result.output
