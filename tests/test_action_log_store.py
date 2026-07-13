from pathlib import Path

import pytest

from mazu.action_log.store import (
    OUTPUT_SUMMARY_MAX_CHARS,
    TOOL_INPUT_MAX_CHARS,
    ActionLogStore,
    record_action,
)


@pytest.fixture
def store(tmp_path: Path) -> ActionLogStore:
    s = ActionLogStore(tmp_path / "action_log.db")
    yield s
    s.close()


def test_log_inserts_and_returns_id(store: ActionLogStore):
    action_id = store.log(
        session_id="s1",
        command="chat",
        tool_name="read_file",
        tool_input='{"path": "a.py"}',
        outcome="ok",
        output_summary="file contents",
        changed_file=None,
    )
    assert action_id > 0


def test_session_actions_orders_chronologically(store: ActionLogStore):
    store.log("s1", "chat", "read_file", "{}", "ok", "first", None)
    store.log("s1", "chat", "write_file", "{}", "ok", "second", "a.py")

    actions = store.session_actions("s1")
    assert [a["output_summary"] for a in actions] == ["first", "second"]


def test_session_actions_filters_by_session(store: ActionLogStore):
    store.log("s1", "chat", "read_file", "{}", "ok", "s1 action", None)
    store.log("s2", "run", "read_file", "{}", "ok", "s2 action", None)

    actions = store.session_actions("s1")
    assert len(actions) == 1
    assert actions[0]["output_summary"] == "s1 action"


def test_session_actions_unknown_session_returns_empty(store: ActionLogStore):
    assert store.session_actions("nope") == []


def test_list_sessions_groups_and_counts(store: ActionLogStore):
    store.log("s1", "chat", "read_file", "{}", "ok", "x", None)
    store.log("s1", "chat", "write_file", "{}", "error", "y", "a.py")
    store.log("s2", "run", "read_file", "{}", "ok", "z", None)

    sessions = {s["session_id"]: s for s in store.list_sessions()}
    assert sessions["s1"]["action_count"] == 2
    assert sessions["s1"]["error_count"] == 1
    assert sessions["s2"]["action_count"] == 1
    assert sessions["s2"]["error_count"] == 0


def test_list_sessions_orders_most_recent_first(store: ActionLogStore):
    store.log("older", "chat", "read_file", "{}", "ok", "x", None)
    store.log("newer", "chat", "read_file", "{}", "ok", "y", None)

    sessions = store.list_sessions()
    assert sessions[0]["session_id"] == "newer"


def test_list_sessions_respects_limit(store: ActionLogStore):
    for i in range(5):
        store.log(f"s{i}", "chat", "read_file", "{}", "ok", "x", None)

    assert len(store.list_sessions(limit=2)) == 2


def test_list_sessions_empty_store(store: ActionLogStore):
    assert store.list_sessions() == []


# ---------------------------------------------------------------------------
# record_action helper
# ---------------------------------------------------------------------------


def test_record_action_writes_a_row(store: ActionLogStore):
    record_action(
        store, "s1", "chat", "read_file", {"path": "a.py"}, "ok", "file contents"
    )
    actions = store.session_actions("s1")
    assert len(actions) == 1
    assert actions[0]["tool_name"] == "read_file"
    assert actions[0]["outcome"] == "ok"
    assert '"path": "a.py"' in actions[0]["tool_input"]


def test_record_action_none_store_is_a_noop():
    # Must not raise -- call sites don't guard every call themselves.
    record_action(None, "s1", "chat", "read_file", {"path": "a.py"}, "ok", "x")


def test_record_action_captures_changed_file_for_write_file(store: ActionLogStore):
    record_action(
        store, "s1", "run", "write_file", {"path": "src/a.py", "content": "x"}, "ok", "Wrote 1 bytes"
    )
    action = store.session_actions("s1")[0]
    assert action["changed_file"] == "src/a.py"


def test_record_action_captures_changed_file_for_edit_file(store: ActionLogStore):
    record_action(
        store, "s1", "run", "edit_file",
        {"path": "src/a.py", "old_str": "x", "new_str": "y"}, "ok", "Edited src/a.py",
    )
    action = store.session_actions("s1")[0]
    assert action["changed_file"] == "src/a.py"


def test_record_action_leaves_changed_file_null_for_non_writing_tools(store: ActionLogStore):
    record_action(store, "s1", "run", "read_file", {"path": "a.py"}, "ok", "contents")
    action = store.session_actions("s1")[0]
    assert action["changed_file"] is None


def test_record_action_leaves_changed_file_null_for_shell(store: ActionLogStore):
    record_action(store, "s1", "run", "run_shell", {"command": "ls"}, "ok", "output")
    action = store.session_actions("s1")[0]
    assert action["changed_file"] is None


def test_record_action_truncates_long_output_summary(store: ActionLogStore):
    huge = "x" * (OUTPUT_SUMMARY_MAX_CHARS + 500)
    record_action(store, "s1", "run", "run_shell", {"command": "ls"}, "ok", huge)
    action = store.session_actions("s1")[0]
    assert len(action["output_summary"]) < len(huge)
    assert "truncated" in action["output_summary"]


def test_record_action_truncates_long_tool_input(store: ActionLogStore):
    huge_content = "x" * (TOOL_INPUT_MAX_CHARS + 500)
    record_action(
        store, "s1", "run", "write_file", {"path": "a.py", "content": huge_content}, "ok", "Wrote bytes"
    )
    action = store.session_actions("s1")[0]
    assert len(action["tool_input"]) < len(huge_content)
    assert "truncated" in action["tool_input"]
    # Even truncated, changed_file resolution reads from the *original* dict, not
    # the truncated serialized string, so it must still be correct.
    assert action["changed_file"] == "a.py"


def test_record_action_handles_non_json_serializable_input_gracefully(store: ActionLogStore):
    class Weird:
        def __str__(self):
            return "weird-object"

    record_action(store, "s1", "run", "read_file", {"path": "a.py", "obj": Weird()}, "ok", "x")
    action = store.session_actions("s1")[0]
    # default=str in json.dumps should stringify the unserializable value rather
    # than raising and crashing the whole tool-call logging path.
    assert "weird-object" in action["tool_input"]
