from datetime import datetime, timedelta, timezone

import pytest

import mazu.curator.loop as curator_loop_module
from mazu.action_log.store import ActionLogStore
from mazu.config import set_config_value
from mazu.curator.orchestrator import run_curator
from mazu.curator.store import CuratorStore
from mazu.llm.types import AgentResponse
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager


def _backdate(store: MemoryStore, memory_id: int, days_ago: int) -> None:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    store.conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (ts, memory_id))
    store.conn.commit()


def _configure_curator():
    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")


def _seed_stale_memory(root) -> int:
    store = MemoryStore(root / ".mazu" / "memory.db")
    memory_id = store.add(category="fact", title="Old fact", body="stale body")
    _backdate(store, memory_id, days_ago=40)
    store.close()
    return memory_id


# ---------------------------------------------------------------------------
# Inertness -- the top-priority guarantee
# ---------------------------------------------------------------------------


def test_unconfigured_curator_run_is_a_complete_noop(tmp_path, monkeypatch):
    def fail_if_called(*a, **k):
        raise AssertionError("should never be called -- Curator is unconfigured")

    monkeypatch.setattr(curator_loop_module, "run_curator_turn", fail_if_called)
    summary = run_curator(tmp_path)
    assert summary.ran is False
    assert summary.reason == "not_configured"
    assert not (tmp_path / ".mazu" / "curator.db").exists()


def test_disabled_curator_run_is_a_complete_noop(tmp_path):
    _configure_curator()
    set_config_value("curator_enabled", "false")
    summary = run_curator(tmp_path)
    assert summary.ran is False
    assert summary.reason == "disabled"
    assert not (tmp_path / ".mazu" / "curator.db").exists()


def test_unknown_area_is_rejected_before_any_store_is_touched(tmp_path):
    _configure_curator()
    summary = run_curator(tmp_path, areas=["bogus_area"])
    assert summary.ran is False
    assert "unknown area" in summary.reason
    assert not (tmp_path / ".mazu" / "curator.db").exists()


# ---------------------------------------------------------------------------
# A real stubbed pass -- proves the tool loop, diary, and "nothing deleted"
# invariant all work end to end without a real API call.
# ---------------------------------------------------------------------------


def _archive_then_stop(memory_id: int):
    """Two fake AgentResponse turns: round 1 calls archive_memory, round 2 ends
    the turn with plain text -- the same shape a real model's response takes."""
    calls = [
        AgentResponse(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": "Archiving the stale one."},
                {
                    "type": "tool_use", "id": "call_1", "name": "archive_memory",
                    "input": {"memory_id": memory_id, "rationale": "unused for 40+ days"},
                },
            ],
            usage={"input_tokens": 100, "output_tokens": 50},
        ),
        AgentResponse(
            stop_reason="end_turn",
            content=[{"type": "text", "text": "Done."}],
            usage={"input_tokens": 20, "output_tokens": 10},
        ),
    ]

    def fake_run_curator_turn(messages, system, tools):
        return calls.pop(0)

    return fake_run_curator_turn


def test_stubbed_pass_archives_memory_and_total_stays_unchanged(tmp_path, monkeypatch):
    _configure_curator()
    memory_id = _seed_stale_memory(tmp_path)
    (tmp_path / ".mazu").mkdir(exist_ok=True)

    store = MemoryStore(tmp_path / ".mazu" / "memory.db")
    before_total = store.stats()["total"]
    store.close()

    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_then_stop(memory_id))

    summary = run_curator(tmp_path)

    assert summary.ran is True
    memory_result = next(r for r in summary.areas if r.area == "memory")
    assert memory_result.ran is True
    assert memory_result.log_entries >= 1

    store = MemoryStore(tmp_path / ".mazu" / "memory.db")
    stats = store.stats()
    active_ids = {r["id"] for r in store.all_active()}
    store.close()

    assert stats["total"] == before_total  # nothing was ever deleted
    assert stats["archived"] == 1
    assert memory_id not in active_ids


def test_dry_run_never_actually_archives(tmp_path, monkeypatch):
    _configure_curator()
    memory_id = _seed_stale_memory(tmp_path)
    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_then_stop(memory_id))

    run_curator(tmp_path, dry_run=True)

    store = MemoryStore(tmp_path / ".mazu" / "memory.db")
    active_ids = {r["id"] for r in store.all_active()}
    stats = store.stats()
    store.close()
    assert memory_id in active_ids
    assert stats["archived"] == 0


def test_diary_entry_has_rationale_and_reversal_hint(tmp_path, monkeypatch):
    _configure_curator()
    memory_id = _seed_stale_memory(tmp_path)
    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_then_stop(memory_id))

    summary = run_curator(tmp_path)

    curator_store = CuratorStore(tmp_path / ".mazu" / "curator.db")
    entries = curator_store.log_for_run(summary.run_id)
    curator_store.close()
    archive_entries = [e for e in entries if e["action"] == "archive_memory"]
    assert len(archive_entries) == 1
    assert archive_entries[0]["rationale"] == "unused for 40+ days"
    assert archive_entries[0]["reversal_hint"] == f"mazu memory unarchive {memory_id}"


def test_second_run_skips_memory_via_watermark_and_costs_nothing(tmp_path, monkeypatch):
    _configure_curator()
    memory_id = _seed_stale_memory(tmp_path)
    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_then_stop(memory_id))
    run_curator(tmp_path)  # first run: archives, advances watermark

    call_count = {"n": 0}

    def fail_if_called(*a, **k):
        call_count["n"] += 1
        raise AssertionError("should not be called -- watermark should have skipped this area")

    monkeypatch.setattr(curator_loop_module, "run_curator_turn", fail_if_called)

    summary = run_curator(tmp_path)  # second run, same day: nothing new to look at

    memory_result = next(r for r in summary.areas if r.area == "memory")
    assert memory_result.ran is False
    assert call_count["n"] == 0
    assert summary.total_cost == 0.0


def test_concurrent_same_area_run_is_serialized_not_racy(tmp_path, monkeypatch):
    """Regression test for a real race found during live multi-process testing:
    without a lock, two concurrent `mazu curator run` invocations on the same
    area could both pass the eligibility check before either advanced the
    watermark, both spending real money redundantly. This simulates a second
    process already holding the area's lock and confirms the orchestrator skips
    cleanly (never crashes, never double-processes) rather than racing."""
    import mazu.curator.orchestrator as orchestrator_module
    from mazu.checkpoint.lock import ReentrantFileLock

    monkeypatch.setattr(
        orchestrator_module, "ReentrantFileLock",
        lambda path: ReentrantFileLock(path, timeout=1.0),
    )
    _configure_curator()
    memory_id = _seed_stale_memory(tmp_path)
    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_then_stop(memory_id))

    lock_path = orchestrator_module._area_lock_path(tmp_path, "memory")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held_lock = ReentrantFileLock(lock_path).acquire()
    held_lock.__enter__()
    try:
        summary = run_curator(tmp_path, areas=["memory"])
    finally:
        held_lock.__exit__(None, None, None)

    memory_result = next(r for r in summary.areas if r.area == "memory")
    assert memory_result.ran is False
    assert "another curator process" in memory_result.skipped_reason

    # Confirm nothing was touched -- the memory is still active, untouched.
    store = MemoryStore(tmp_path / ".mazu" / "memory.db")
    active_ids = {r["id"] for r in store.all_active()}
    store.close()
    assert memory_id in active_ids


def test_forced_area_ignores_watermark(tmp_path, monkeypatch):
    _configure_curator()
    memory_id = _seed_stale_memory(tmp_path)
    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_then_stop(memory_id))
    run_curator(tmp_path)  # advances the watermark

    # Seed a second stale memory so there's something for the forced pass to do.
    store = MemoryStore(tmp_path / ".mazu" / "memory.db")
    second_id = store.add(category="fact", title="Another old fact", body="also stale")
    _backdate(store, second_id, days_ago=40)
    store.close()

    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_then_stop(second_id))

    summary = run_curator(tmp_path, areas=["memory"])  # --area memory forces regardless of watermark

    memory_result = next(r for r in summary.areas if r.area == "memory")
    assert memory_result.ran is True


# ---------------------------------------------------------------------------
# Skills area
# ---------------------------------------------------------------------------


def _archive_skill_then_stop(skill_name: str):
    calls = [
        AgentResponse(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": "Archiving the failing skill."},
                {
                    "type": "tool_use", "id": "call_1", "name": "archive_skill",
                    "input": {"name": skill_name, "rationale": "fails most of the time"},
                },
            ],
            usage={"input_tokens": 100, "output_tokens": 50},
        ),
        AgentResponse(
            stop_reason="end_turn", content=[{"type": "text", "text": "Done."}],
            usage={"input_tokens": 20, "output_tokens": 10},
        ),
    ]

    def fake_run_curator_turn(messages, system, tools):
        return calls.pop(0)

    return fake_run_curator_turn


def _seed_failing_skill(root) -> str:
    manager = SkillManager(root)
    manager.save("flaky_skill", "does something", "def run(args):\n    return 'ok'\n")
    action_log = ActionLogStore(root / ".mazu" / "action_log.db")
    for outcome in ("error", "error", "error", "ok"):
        action_log.log("s1", "chat", "run_skill", '{"name": "flaky_skill", "args": {}}', outcome, "out", None)
    action_log.close()
    return "flaky_skill"


def test_skills_area_archives_a_failing_skill(tmp_path, monkeypatch):
    _configure_curator()
    name = _seed_failing_skill(tmp_path)
    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_skill_then_stop(name))

    summary = run_curator(tmp_path, areas=["skills"])

    assert summary.ran is True
    skills_result = next(r for r in summary.areas if r.area == "skills")
    assert skills_result.ran is True

    manager = SkillManager(tmp_path)
    assert manager.list() == []  # archived, no longer active
    meta = manager.get_meta(name)
    assert meta["archived"] is True
    assert manager.exists(name) is True  # still on disk, reversible


def test_skills_area_not_eligible_without_signal(tmp_path, monkeypatch):
    """A healthy skill (no meaningful failure rate) should never trigger a
    default (non-forced) pass -- proves the zero-cost eligibility sweep actually
    gates skills the same way it gates memory."""
    _configure_curator()
    manager = SkillManager(tmp_path)
    manager.save("healthy_skill", "works fine", "def run(args):\n    return 'ok'\n")
    action_log = ActionLogStore(tmp_path / ".mazu" / "action_log.db")
    action_log.log("s1", "chat", "run_skill", '{"name": "healthy_skill"}', "ok", "out", None)
    action_log.close()

    def fail_if_called(*a, **k):
        raise AssertionError("should not be called -- no real signal for skills area")

    monkeypatch.setattr(curator_loop_module, "run_curator_turn", fail_if_called)

    summary = run_curator(tmp_path)  # default areas, not forced
    skills_result = next(r for r in summary.areas if r.area == "skills")
    assert skills_result.ran is False


# ---------------------------------------------------------------------------
# Housekeeping area
# ---------------------------------------------------------------------------


def _set_config_then_stop(key: str, value: str):
    calls = [
        AgentResponse(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": "Tuning config."},
                {
                    "type": "tool_use", "id": "call_1", "name": "set_config",
                    "input": {"key": key, "value": value, "rationale": "evidence-based tuning"},
                },
            ],
            usage={"input_tokens": 100, "output_tokens": 50},
        ),
        AgentResponse(
            stop_reason="end_turn", content=[{"type": "text", "text": "Done."}],
            usage={"input_tokens": 20, "output_tokens": 10},
        ),
    ]

    def fake_run_curator_turn(messages, system, tools):
        return calls.pop(0)

    return fake_run_curator_turn


def test_housekeeping_area_can_tune_an_allowlisted_config_key(tmp_path, monkeypatch):
    from mazu.action_log.store import ActionLogStore as _ActionLogStore

    _configure_curator()
    # Give the zero-cost eligibility sweep a real signal: enough action-log volume.
    action_log = _ActionLogStore(tmp_path / ".mazu" / "action_log.db")
    for i in range(12):
        action_log.log("s1", "chat", "read_file", "{}", "ok", "x", None)
    action_log.close()

    monkeypatch.setattr(
        curator_loop_module, "run_curator_turn",
        _set_config_then_stop("router_suggestions", "false"),
    )

    summary = run_curator(tmp_path, areas=["housekeeping"])

    assert summary.ran is True
    housekeeping_result = next(r for r in summary.areas if r.area == "housekeeping")
    assert housekeeping_result.ran is True

    from mazu.config import list_config
    assert list_config()["router_suggestions"] == "false"


# ---------------------------------------------------------------------------
# Conflicts area
# ---------------------------------------------------------------------------


def _record_and_resolve_conflict_then_stop(id_a: int, id_b: int):
    calls = [
        AgentResponse(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": "These contradict."},
                {
                    "type": "tool_use", "id": "call_1", "name": "record_conflict",
                    "input": {
                        "memory_id_a": id_a, "memory_id_b": id_b, "kind": "contradiction",
                        "rationale": "they disagree about which database was chosen",
                    },
                },
            ],
            usage={"input_tokens": 100, "output_tokens": 50},
        ),
        AgentResponse(
            stop_reason="tool_use",
            content=[
                {
                    "type": "tool_use", "id": "call_2", "name": "supersede_memory",
                    "input": {"old_id": id_a, "new_id": id_b, "rationale": "id_b is the current decision"},
                },
            ],
            usage={"input_tokens": 50, "output_tokens": 30},
        ),
        AgentResponse(
            stop_reason="end_turn", content=[{"type": "text", "text": "Done."}],
            usage={"input_tokens": 20, "output_tokens": 10},
        ),
    ]

    def fake_run_curator_turn(messages, system, tools):
        return calls.pop(0)

    return fake_run_curator_turn


def _seed_conflicting_memories(root) -> tuple[int, int]:
    store = MemoryStore(root / ".mazu" / "memory.db")
    id_a = store.add(
        category="decision", title="use postgres for storage",
        body="chose postgres because of jsonb support and strong community tooling",
    )
    id_b = store.add(
        category="decision", title="use mysql for storage",
        body="chose mysql for the team familiarity and existing ops tooling",
    )
    store.close()
    return id_a, id_b


def test_conflicts_area_records_and_resolves_a_contradiction(tmp_path, monkeypatch):
    _configure_curator()
    id_a, id_b = _seed_conflicting_memories(tmp_path)
    monkeypatch.setattr(
        curator_loop_module, "run_curator_turn", _record_and_resolve_conflict_then_stop(id_a, id_b)
    )

    summary = run_curator(tmp_path, areas=["conflicts"])

    assert summary.ran is True
    conflicts_result = next(r for r in summary.areas if r.area == "conflicts")
    assert conflicts_result.ran is True

    curator_store = CuratorStore(tmp_path / ".mazu" / "curator.db")
    conflicts = curator_store.list_conflicts(unresolved_only=False)
    curator_store.close()
    assert len(conflicts) == 1
    assert conflicts[0]["kind"] == "contradiction"

    store = MemoryStore(tmp_path / ".mazu" / "memory.db")
    active_ids = {r["id"] for r in store.all_active()}
    stats = store.stats()
    store.close()
    assert id_a not in active_ids  # superseded, not deleted
    assert id_b in active_ids
    assert stats["total"] == 2  # nothing permanently removed


# ---------------------------------------------------------------------------
# Run-level status accuracy when an area errors
# ---------------------------------------------------------------------------


def test_run_status_reflects_an_area_that_errored(tmp_path, monkeypatch):
    """A run where every requested area either ran cleanly or was legitimately
    skipped must be 'completed'; a run where at least one area genuinely errored
    (a real API/tool exception) must say so in curator_runs.status, not silently
    report 'completed' while burying the failure in one AreaResult."""
    _configure_curator()
    _seed_stale_memory(tmp_path)

    def _raise(*a, **k):
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _raise)

    summary = run_curator(tmp_path, areas=["memory"])

    memory_result = next(r for r in summary.areas if r.area == "memory")
    assert memory_result.ran is False
    assert memory_result.skipped_reason.startswith("error:")

    curator_store = CuratorStore(tmp_path / ".mazu" / "curator.db")
    run = curator_store.last_run(summary.run_id)
    curator_store.close()
    assert run["status"] == "completed_with_errors"


def test_run_status_is_plain_completed_when_nothing_errors(tmp_path, monkeypatch):
    _configure_curator()
    memory_id = _seed_stale_memory(tmp_path)
    monkeypatch.setattr(curator_loop_module, "run_curator_turn", _archive_then_stop(memory_id))

    summary = run_curator(tmp_path, areas=["memory"])

    curator_store = CuratorStore(tmp_path / ".mazu" / "curator.db")
    run = curator_store.last_run(summary.run_id)
    curator_store.close()
    assert run["status"] == "completed"
