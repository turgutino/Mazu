"""Tests for `mazu memory-belief stats`/`review` -- the read-only inspection
tool over belief_shadow.db (Phase 3's shadow-mode evidence log). These commands
must never write to memory.db, never approve/reject/apply anything, and never
fabricate a confident-looking result out of too little data.
"""
import subprocess

import pytest
from click.testing import CliRunner

from mazu.cli import _belief_shadow_db_path, _memory_db_path, main
from mazu.memory.belief_log import BeliefShadowLogStore
from mazu.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])
    monkeypatch.chdir(tmp_path)


def _seed_log(root, decisions):
    log = BeliefShadowLogStore(_belief_shadow_db_path(root))
    for d in decisions:
        log.log_decision(**d)
    log.close()


def _decision(**overrides):
    base = dict(
        session_id="s1",
        category="fact",
        new_title="Project now uses SQLite",
        new_body="for local dev",
        related_memory_id=1,
        related_title="Project uses PostgreSQL",
        similarity_score=0.3,
        relationship="replacement",
        confidence="high",
        proposed_action="supersede_old_with_new",
        would_supersede=True,
        classifier_failed=False,
        reasoning="clean architectural swap",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_on_empty_database(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "stats"])
    assert result.exit_code == 0
    assert "Total shadow decisions: 0" in result.output
    assert "nothing to evaluate" in result.output.lower()


def test_stats_below_min_decisions_shows_caveat(tmp_path):
    _seed_log(tmp_path, [_decision() for _ in range(3)])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "stats"])
    assert result.exit_code == 0
    assert "Total shadow decisions: 3" in result.output
    assert "too small a sample" in result.output.lower()


def test_stats_reports_full_distribution_with_enough_data(tmp_path):
    decisions = (
        [_decision(relationship="replacement", confidence="high", proposed_action="supersede_old_with_new", would_supersede=True) for _ in range(8)]
        + [_decision(relationship="contradiction", confidence="medium", proposed_action="flag_for_review", would_supersede=False) for _ in range(7)]
        + [_decision(relationship="duplicate", confidence="high", proposed_action="skip_duplicate", would_supersede=False) for _ in range(5)]
        + [_decision(relationship=None, confidence=None, proposed_action="classifier_failed", would_supersede=False, classifier_failed=True, reasoning="classifier error: timeout")]
    )
    _seed_log(tmp_path, decisions)

    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "stats"])
    assert result.exit_code == 0
    assert "Total shadow decisions: 21" in result.output
    assert "too small a sample" not in result.output.lower()
    assert "replacement: 8" in result.output
    assert "contradiction: 7" in result.output
    assert "duplicate: 5" in result.output
    assert "Classifier failures: 1" in result.output
    assert "Would have superseded an old memory: 8" in result.output
    assert "Flagged for review (not auto-resolved): 7" in result.output
    assert "SHADOW MODE" in result.output


def test_stats_json_output(tmp_path):
    _seed_log(tmp_path, [_decision(similarity_score=0.2), _decision(similarity_score=0.4)])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "stats", "--json"])
    assert result.exit_code == 0
    import json

    data = json.loads(result.output)["data"]
    assert data["total"] == 2
    assert data["avg_similarity_score"] == pytest.approx(0.3)
    assert data["would_supersede_count"] == 2


def test_stats_average_similarity_ignores_null_scores(tmp_path):
    _seed_log(tmp_path, [_decision(similarity_score=0.5), _decision(similarity_score=None, relationship=None, confidence=None, classifier_failed=True, proposed_action="classifier_failed")])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "stats"])
    assert "Average similarity score: 0.500" in result.output


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


def test_review_on_empty_database(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review"])
    assert result.exit_code == 0
    assert "No shadow decisions match" in result.output


def test_review_shows_all_required_fields(tmp_path):
    _seed_log(tmp_path, [_decision(
        new_title="Auth uses OAuth2", new_body="delegated auth",
        related_title="Auth uses JWT", similarity_score=0.25,
        relationship="contradiction", confidence="medium", reasoning="ambiguous from text alone",
        proposed_action="flag_for_review", would_supersede=False,
    )])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review"])
    assert result.exit_code == 0
    assert "Auth uses OAuth2" in result.output
    assert "delegated auth" in result.output
    assert "Auth uses JWT" in result.output
    assert "0.250" in result.output
    assert "contradiction" in result.output
    assert "medium" in result.output
    assert "flag_for_review" in result.output
    assert "ambiguous from text alone" in result.output
    assert "shadow mode" in result.output.lower()


def test_review_marks_classifier_failure_distinctly(tmp_path):
    _seed_log(tmp_path, [_decision(
        relationship=None, confidence=None, proposed_action="classifier_failed",
        classifier_failed=True, reasoning="classifier error: API down", would_supersede=False,
    )])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review"])
    assert "CLASSIFIER FAILED" in result.output
    assert "API down" in result.output


def test_review_filter_by_relationship(tmp_path):
    _seed_log(tmp_path, [
        _decision(new_title="Marker Contradiction Item", relationship="contradiction"),
        _decision(new_title="Marker Duplicate Item", relationship="duplicate"),
    ])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review", "--relationship", "duplicate"])
    assert "Marker Duplicate Item" in result.output
    assert "Marker Contradiction Item" not in result.output


def test_review_filter_by_confidence(tmp_path):
    _seed_log(tmp_path, [
        _decision(new_title="Marker Low Item", confidence="low"),
        _decision(new_title="Marker High Item", confidence="high"),
    ])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review", "--confidence", "high"])
    assert "Marker High Item" in result.output
    assert "Marker Low Item" not in result.output


def test_review_combined_filters(tmp_path):
    _seed_log(tmp_path, [
        _decision(new_title="Marker Contra High", relationship="contradiction", confidence="high"),
        _decision(new_title="Marker Contra Low", relationship="contradiction", confidence="low"),
        _decision(new_title="Marker Dup High", relationship="duplicate", confidence="high"),
    ])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review", "--relationship", "contradiction", "--confidence", "high"])
    assert "Marker Contra High" in result.output
    assert "Marker Contra Low" not in result.output
    assert "Marker Dup High" not in result.output


def test_review_respects_limit(tmp_path):
    _seed_log(tmp_path, [_decision(new_title=f"item-{i:03d}") for i in range(10)])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review", "--limit", "3"])
    assert result.output.count("NEW:") == 3


def test_review_default_limit_is_20(tmp_path):
    _seed_log(tmp_path, [_decision(new_title=f"item-{i:03d}") for i in range(25)])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review"])
    assert result.output.count("NEW:") == 20


def test_review_most_recent_first(tmp_path):
    import time

    log = BeliefShadowLogStore(_belief_shadow_db_path(tmp_path))
    for i in range(3):
        log.log_decision(**_decision(new_title=f"item-{i}"))
        time.sleep(0.01)
    log.close()

    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review", "--limit", "1"])
    assert "item-2" in result.output
    assert "item-0" not in result.output


def test_review_json_output(tmp_path):
    _seed_log(tmp_path, [_decision(new_title="Only one")])
    runner = CliRunner()
    result = runner.invoke(main, ["memory-belief", "review", "--json"])
    import json

    data = json.loads(result.output)["data"]
    assert len(data) == 1
    assert data[0]["new_title"] == "Only one"
    assert isinstance(data[0]["would_supersede"], bool)
    assert isinstance(data[0]["classifier_failed"], bool)


# ---------------------------------------------------------------------------
# Multiple projects / databases
# ---------------------------------------------------------------------------


def test_stats_is_scoped_to_the_current_project_only(tmp_path, monkeypatch):
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    _seed_log(project_a, [_decision(new_title="only in A") for _ in range(5)])
    _seed_log(project_b, [_decision(new_title="only in B") for _ in range(2)])

    runner = CliRunner()
    monkeypatch.chdir(project_a)
    result_a = runner.invoke(main, ["memory-belief", "stats"])
    assert "Total shadow decisions: 5" in result_a.output

    monkeypatch.chdir(project_b)
    result_b = runner.invoke(main, ["memory-belief", "stats"])
    assert "Total shadow decisions: 2" in result_b.output


# ---------------------------------------------------------------------------
# Never mutates memory.db
# ---------------------------------------------------------------------------


def test_stats_never_touches_memory_db(tmp_path):
    store = MemoryStore(_memory_db_path(tmp_path))
    mem_id = store.add(category="fact", title="Untouched fact", body="should stay exactly as-is")
    store.close()

    _seed_log(tmp_path, [_decision(relationship="replacement", confidence="high", would_supersede=True) for _ in range(25)])

    runner = CliRunner()
    runner.invoke(main, ["memory-belief", "stats"])

    check = MemoryStore(_memory_db_path(tmp_path))
    row = check.get(mem_id)
    check.close()
    assert row["title"] == "Untouched fact"
    assert row["superseded_by"] is None
    assert row["needs_review"] == 0


def test_review_never_touches_memory_db(tmp_path):
    store = MemoryStore(_memory_db_path(tmp_path))
    mem_id = store.add(category="fact", title="Untouched fact", body="should stay exactly as-is")
    store.close()

    _seed_log(tmp_path, [_decision(relationship="contradiction", confidence="high")])

    runner = CliRunner()
    runner.invoke(main, ["memory-belief", "review"])

    check = MemoryStore(_memory_db_path(tmp_path))
    row = check.get(mem_id)
    check.close()
    assert row["title"] == "Untouched fact"
    assert row["superseded_by"] is None
    assert row["needs_review"] == 0


def test_memory_db_not_even_created_by_belief_commands(tmp_path):
    """If a project has shadow decisions but no memory.db yet at all, these
    commands must not create one -- they have no business touching it."""
    _seed_log(tmp_path, [_decision()])
    runner = CliRunner()
    runner.invoke(main, ["memory-belief", "stats"])
    runner.invoke(main, ["memory-belief", "review"])
    assert not _memory_db_path(tmp_path).exists()
