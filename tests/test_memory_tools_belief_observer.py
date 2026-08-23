"""Tests for the SHADOW-ONLY observer attached to the real `remember` tool
(mazu/tools/memory_tools.py). This observer records real, explicit
insert/supersede actions the agent takes -- never an LLM call, never a change
to remember's own behavior. See mazu/memory/belief_log.py's
explicit_memory_actions table.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from mazu.memory.belief_log import BeliefShadowLogStore
from mazu.memory.store import MemoryStore
from mazu.tools.memory_tools import make_memory_tools


@pytest.fixture
def stores(tmp_path: Path):
    project = MemoryStore(tmp_path / "project.db")
    glob = MemoryStore(tmp_path / "global.db")
    yield project, glob
    project.close()
    glob.close()


@pytest.fixture
def belief_log(tmp_path: Path) -> BeliefShadowLogStore:
    log = BeliefShadowLogStore(tmp_path / "belief_shadow.db")
    yield log
    log.close()


def _tools(stores, belief_log=None, project="proj"):
    p, glob = stores
    return {t.name: t for t in make_memory_tools(p, glob, session_id="s1", belief_log=belief_log, project=project)}


# ---------------------------------------------------------------------------
# 1 & 2: remember/supersede behave exactly as before, with or without belief_log
# ---------------------------------------------------------------------------


def test_remember_plain_insert_unchanged_without_belief_log(stores):
    tools = _tools(stores, belief_log=None)
    result = tools["remember"].handler(
        {"category": "fact", "title": "Uses SQLite", "body": "for local dev"}
    )
    assert not result.is_error
    project, _ = stores
    assert len(project.all_active()) == 1


def test_remember_plain_insert_unchanged_with_belief_log(stores, belief_log):
    tools = _tools(stores, belief_log=belief_log)
    result = tools["remember"].handler(
        {"category": "fact", "title": "Uses SQLite", "body": "for local dev"}
    )
    assert not result.is_error
    project, _ = stores
    assert len(project.all_active()) == 1
    row = project.all_active()[0]
    assert row["title"] == "Uses SQLite"
    assert row["body"] == "for local dev"


def test_remember_supersede_unchanged_with_belief_log(stores, belief_log):
    project, _ = stores
    old_id = project.add(category="decision", title="Use MySQL", body="initial pick")

    tools = _tools(stores, belief_log=belief_log)
    result = tools["remember"].handler(
        {
            "category": "decision",
            "title": "Use PostgreSQL",
            "body": "switched for concurrency",
            "supersedes_id": old_id,
        }
    )
    assert not result.is_error
    assert "marked memory" in result.content
    assert old_id not in {r["id"] for r in project.all_active()}
    old_row = project.get(old_id)
    assert old_row["superseded_by"] is not None
    assert old_row["title"] == "Use MySQL"  # content untouched, only retired


def test_remember_output_message_identical_with_and_without_belief_log(stores, belief_log):
    tools_no_log = _tools(stores, belief_log=None)
    result_no_log = tools_no_log["remember"].handler(
        {"category": "fact", "title": "A", "body": "B"}
    )

    stores2 = (MemoryStore(stores[0].db_path.parent / "project2.db"), MemoryStore(stores[1].db_path.parent / "global2.db"))
    tools_with_log = _tools(stores2, belief_log=belief_log)
    result_with_log = tools_with_log["remember"].handler(
        {"category": "fact", "title": "A", "body": "B"}
    )
    stores2[0].close()
    stores2[1].close()

    # Same input, same category of message shape (ids will differ, but the
    # template/behavior is identical) -- confirms belief_log presence changes
    # nothing about what remember reports back to the agent.
    assert result_no_log.content.startswith("Remembered (id=")
    assert result_with_log.content.startswith("Remembered (id=")
    assert not result_no_log.is_error
    assert not result_with_log.is_error


# ---------------------------------------------------------------------------
# 3: shadow observation records both actions
# ---------------------------------------------------------------------------


def test_plain_insert_is_recorded_as_explicit_action(stores, belief_log):
    tools = _tools(stores, belief_log=belief_log, project="my-project")
    tools["remember"].handler({"category": "fact", "title": "Uses SQLite", "body": "for local dev"})

    actions = belief_log.all_explicit_actions()
    assert len(actions) == 1
    row = actions[0]
    assert row["action_type"] == "insert"
    assert row["session_id"] == "s1"
    assert row["project"] == "my-project"
    assert row["category"] == "fact"
    assert row["new_title"] == "Uses SQLite"
    assert row["new_body"] == "for local dev"
    assert row["old_memory_id"] is None
    assert row["asserted_relationship"] is None
    assert row["tool_success"] == 1


def test_supersede_is_recorded_as_explicit_action_with_asserted_relationship(stores, belief_log):
    project, _ = stores
    old_id = project.add(category="decision", title="Use MySQL", body="initial pick, no replication")

    tools = _tools(stores, belief_log=belief_log)
    tools["remember"].handler(
        {
            "category": "decision",
            "title": "Use PostgreSQL",
            "body": "switched for concurrency and replication",
            "supersedes_id": old_id,
        }
    )

    actions = belief_log.all_explicit_actions()
    assert len(actions) == 1
    row = actions[0]
    assert row["action_type"] == "supersede"
    assert row["old_memory_id"] == old_id
    assert row["old_title"] == "Use MySQL"
    assert row["old_body"] == "initial pick, no replication"
    assert row["new_title"] == "Use PostgreSQL"
    # The relationship is ASSERTED by the agent's own action, never guessed by a classifier.
    assert row["asserted_relationship"] == "replacement"
    assert row["similarity_score"] is not None
    assert row["tool_success"] == 1


def test_insert_near_an_existing_similar_memory_notes_the_nearest_active_one(stores, belief_log):
    project, _ = stores
    project.add(category="fact", title="CI runs on GitHub Actions", body="unrelated tooling fact")

    tools = _tools(stores, belief_log=belief_log)
    tools["remember"].handler(
        {"category": "fact", "title": "Project uses SQLite database", "body": "This project's database is SQLite."}
    )

    actions = belief_log.all_explicit_actions()
    assert len(actions) == 1
    row = actions[0]
    assert row["action_type"] == "insert"
    # Whether or not a gray-zone match is found depends on lexical overlap --
    # this test only asserts the field is populated consistently (present or
    # both None together), not a specific similarity value (that would depend
    # on the exact Jaccard threshold behavior already tested elsewhere).
    if row["nearest_active_memory_id"] is not None:
        assert row["nearest_active_similarity"] is not None
    else:
        assert row["nearest_active_similarity"] is None


def test_insert_with_no_similar_existing_memory_has_no_nearest_match(stores, belief_log):
    tools = _tools(stores, belief_log=belief_log)
    tools["remember"].handler({"category": "fact", "title": "Totally novel fact", "body": "nothing like anything else"})

    row = belief_log.all_explicit_actions()[0]
    assert row["nearest_active_memory_id"] is None
    assert row["nearest_active_similarity"] is None


def test_failed_supersede_target_still_records_the_insert_half(stores, belief_log):
    """supersedes_id pointing at a nonexistent memory: the insert still succeeds
    (existing behavior, unchanged) and the shadow observer still records it as a
    supersede attempt, with the real tool_result text carrying the warning."""
    tools = _tools(stores, belief_log=belief_log)
    result = tools["remember"].handler(
        {"category": "decision", "title": "New decision", "body": "body", "supersedes_id": 9999}
    )
    assert not result.is_error
    assert "warning" in result.content

    row = belief_log.all_explicit_actions()[0]
    assert row["action_type"] == "supersede"
    assert row["old_memory_id"] == 9999
    assert row["old_title"] is None  # nothing to fetch -- never fabricated
    assert "warning" in row["tool_result"]


# ---------------------------------------------------------------------------
# 4: shadow logging failure never blocks the real action
# ---------------------------------------------------------------------------


def test_belief_log_failure_never_blocks_a_real_insert(stores, belief_log):
    tools = _tools(stores, belief_log=belief_log)
    with patch.object(BeliefShadowLogStore, "log_explicit_action", side_effect=RuntimeError("disk full")):
        result = tools["remember"].handler({"category": "fact", "title": "Still saved", "body": "body"})

    assert not result.is_error
    assert "Remembered" in result.content
    project, _ = stores
    assert len(project.all_active()) == 1
    assert project.all_active()[0]["title"] == "Still saved"


def test_belief_log_failure_never_blocks_a_real_supersede(stores, belief_log):
    project, _ = stores
    old_id = project.add(category="decision", title="Old", body="old body")

    tools = _tools(stores, belief_log=belief_log)
    with patch.object(BeliefShadowLogStore, "log_explicit_action", side_effect=RuntimeError("disk full")):
        result = tools["remember"].handler(
            {"category": "decision", "title": "New", "body": "new body", "supersedes_id": old_id}
        )

    assert not result.is_error
    assert "marked memory" in result.content
    assert project.get(old_id)["superseded_by"] is not None


def test_find_gray_zone_match_failure_never_blocks_a_real_insert(stores, belief_log):
    tools = _tools(stores, belief_log=belief_log)
    with patch("mazu.memory.belief.find_gray_zone_match", side_effect=RuntimeError("boom")):
        result = tools["remember"].handler({"category": "fact", "title": "Still saved 2", "body": "body"})

    assert not result.is_error
    project, _ = stores
    assert len(project.all_active()) == 1


# ---------------------------------------------------------------------------
# 5: no additional LLM call occurs merely because logging is enabled
# ---------------------------------------------------------------------------


def test_no_classifier_call_for_plain_insert(stores, belief_log):
    tools = _tools(stores, belief_log=belief_log)
    with patch("mazu.memory.belief.classify_relationship") as mock_classify:
        tools["remember"].handler({"category": "fact", "title": "New fact", "body": "body"})
    mock_classify.assert_not_called()


def test_no_classifier_call_for_supersede(stores, belief_log):
    project, _ = stores
    old_id = project.add(category="decision", title="Old", body="old body")
    tools = _tools(stores, belief_log=belief_log)
    with patch("mazu.memory.belief.classify_relationship") as mock_classify:
        tools["remember"].handler(
            {"category": "decision", "title": "New", "body": "new body", "supersedes_id": old_id}
        )
    mock_classify.assert_not_called()


def test_no_run_forced_tool_call_at_all_from_remember(stores, belief_log):
    """Even more directly: remember() must never reach the LLM client layer,
    with or without a gray-zone match, with or without belief_log."""
    project, _ = stores
    project.add(category="fact", title="CI runs on GitHub Actions", body="unrelated tooling fact")
    tools = _tools(stores, belief_log=belief_log)
    with patch("mazu.llm.client.run_forced_tool") as mock_llm:
        tools["remember"].handler(
            {"category": "fact", "title": "Project uses SQLite database", "body": "database is SQLite"}
        )
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# 6: existing auto-extraction shadow behavior remains unchanged
# ---------------------------------------------------------------------------


def test_auto_extraction_shadow_path_unaffected_by_explicit_observer(stores, belief_log):
    """The Phase 3 auto-extraction shadow batch (mazu.memory.belief.run_shadow_belief_batch)
    and the new explicit-action observer write to different tables and must not
    interfere with each other."""
    from mazu.memory.belief import run_shadow_belief_batch

    project, _ = stores
    tools = _tools(stores, belief_log=belief_log)
    tools["remember"].handler({"category": "fact", "title": "Explicit fact", "body": "body one"})

    with patch(
        "mazu.memory.belief.classify_relationship",
        return_value=({"relationship": "unrelated", "confidence": "high", "reasoning": "n/a"}, {}),
    ):
        run_shadow_belief_batch(
            project, belief_log, "s1",
            [{"category": "fact", "title": "Auto-extracted fact", "body": "totally different content here"}],
        )

    assert len(belief_log.all_explicit_actions()) == 1
    assert len(belief_log.all_decisions()) <= 1  # only if it happened to land in gray zone
    # The two tables are independent -- explicit action count is unaffected by
    # whatever the auto-extraction shadow batch did.
    assert belief_log.all_explicit_actions()[0]["new_title"] == "Explicit fact"


# ---------------------------------------------------------------------------
# 7: project/session isolation remains correct
# ---------------------------------------------------------------------------


def test_explicit_actions_are_scoped_by_project_field(stores, belief_log):
    tools_a = _tools(stores, belief_log=belief_log, project="/path/to/project-a")
    tools_a["remember"].handler({"category": "fact", "title": "Fact in A", "body": "body"})

    stores_b = (MemoryStore(stores[0].db_path.parent / "b_project.db"), stores[1])
    tools_b = _tools(stores_b, belief_log=belief_log, project="/path/to/project-b")
    tools_b["remember"].handler({"category": "fact", "title": "Fact in B", "body": "body"})
    stores_b[0].close()

    actions = belief_log.all_explicit_actions()
    assert len(actions) == 2
    projects = {a["project"] for a in actions}
    assert projects == {"/path/to/project-a", "/path/to/project-b"}


def test_explicit_actions_carry_the_correct_session_id(stores, belief_log):
    p, glob = stores
    tools_s1 = {t.name: t for t in make_memory_tools(p, glob, session_id="session-1", belief_log=belief_log)}
    tools_s1["remember"].handler({"category": "fact", "title": "From session 1", "body": "body"})

    tools_s2 = {t.name: t for t in make_memory_tools(p, glob, session_id="session-2", belief_log=belief_log)}
    tools_s2["remember"].handler({"category": "fact", "title": "From session 2", "body": "body"})

    actions = {a["new_title"]: a["session_id"] for a in belief_log.all_explicit_actions()}
    assert actions["From session 1"] == "session-1"
    assert actions["From session 2"] == "session-2"


def test_user_preference_insert_still_routes_to_global_store_with_observer_attached(stores, belief_log):
    tools = _tools(stores, belief_log=belief_log)
    result = tools["remember"].handler({"category": "user_preference", "title": "Name", "body": "Turgut"})
    assert not result.is_error

    project, glob = stores
    assert len(project.all_active()) == 0
    assert len(glob.all_active()) == 1

    row = belief_log.all_explicit_actions()[0]
    assert row["category"] == "user_preference"
