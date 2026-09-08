from pathlib import Path

import pytest

from mazu.chat.store import ChatStore


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    s = ChatStore(tmp_path / "chat_history.db")
    yield s
    s.close()


def test_append_message_creates_the_session_on_first_call(store: ChatStore):
    store.append_message("s1", "user", "fix the parser bug", model="deepseek:deepseek-chat")
    session = store.get_session("s1")
    assert session is not None
    assert session["model"] == "deepseek:deepseek-chat"
    assert session["title"] == "fix the parser bug"


def test_append_message_updates_model_on_later_calls_not_just_the_first(store: ChatStore):
    # Real bug found live: a mid-conversation model switch (the web Chat page's
    # model chip, or ChatSession.set_model()) passes the NEW model into every
    # later append_message() call, but the column only ever got set on the
    # first INSERT -- so a resumed session always reverted to whichever model
    # the conversation started with, silently discarding a later switch.
    store.append_message("s1", "user", "hello", model="deepseek:deepseek-chat")
    store.append_message("s1", "assistant", [{"type": "text", "text": "hi"}], model="deepseek:deepseek-chat")
    store.append_message("s1", "user", "switching models now", model="openai:gpt-4o")
    assert store.get_session("s1")["model"] == "openai:gpt-4o"


def test_append_message_without_a_model_leaves_the_stored_model_unchanged(store: ChatStore):
    store.append_message("s1", "user", "hello", model="deepseek:deepseek-chat")
    store.append_message("s1", "assistant", [{"type": "text", "text": "hi"}])
    assert store.get_session("s1")["model"] == "deepseek:deepseek-chat"


def test_append_message_never_overwrites_the_title_after_the_first_user_message(store: ChatStore):
    store.append_message("s1", "user", "first question")
    store.append_message("s1", "assistant", [{"type": "text", "text": "answer"}])
    store.append_message("s1", "user", "second question")
    assert store.get_session("s1")["title"] == "first question"


def test_get_messages_preserves_order_and_round_trips_content_shapes(store: ChatStore):
    store.append_message("s1", "user", "hello")
    store.append_message("s1", "assistant", [{"type": "text", "text": "hi there"}])
    store.append_message("s1", "user", [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}])

    messages = store.get_messages("s1")
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]},
    ]


def test_get_messages_for_unknown_session_returns_empty_list(store: ChatStore):
    assert store.get_messages("nonexistent") == []


def test_list_sessions_orders_most_recently_updated_first_with_live_message_count(store: ChatStore):
    store.append_message("s1", "user", "first session")
    store.append_message("s2", "user", "second session")
    store.append_message("s2", "assistant", [{"type": "text", "text": "reply"}])
    # Touch s1 again so it becomes the most recently updated.
    store.append_message("s1", "assistant", [{"type": "text", "text": "reply to s1"}])

    sessions = store.list_sessions()
    assert [s["session_id"] for s in sessions] == ["s1", "s2"]
    assert [s["message_count"] for s in sessions] == [2, 2]


def test_list_sessions_respects_limit(store: ChatStore):
    for i in range(5):
        store.append_message(f"s{i}", "user", f"session {i}")
    assert len(store.list_sessions(limit=2)) == 2


def test_latest_session_id_returns_none_when_store_is_empty(store: ChatStore):
    assert store.latest_session_id() is None


def test_latest_session_id_returns_the_most_recently_updated_session(store: ChatStore):
    store.append_message("s1", "user", "old")
    store.append_message("s2", "user", "new")
    assert store.latest_session_id() == "s2"


def test_delete_session_removes_the_session_and_its_messages(store: ChatStore):
    store.append_message("s1", "user", "hello")
    store.append_message("s1", "assistant", [{"type": "text", "text": "hi"}])

    ok = store.delete_session("s1")
    assert ok is True
    assert store.get_session("s1") is None
    assert store.get_messages("s1") == []


def test_delete_session_does_not_touch_other_sessions(store: ChatStore):
    store.append_message("s1", "user", "keep me")
    store.append_message("s2", "user", "delete me")

    store.delete_session("s2")
    assert store.get_session("s1") is not None
    assert store.get_messages("s1") == [{"role": "user", "content": "keep me"}]


def test_delete_session_returns_false_for_an_unknown_session(store: ChatStore):
    assert store.delete_session("nonexistent") is False


def test_a_message_with_no_role_change_still_gets_increasing_seq_numbers(store: ChatStore):
    # Not exposed directly, but get_messages()'s ordering depends on seq being
    # monotonically increasing per session even across many appends -- this is the
    # actual guarantee that makes replay order-correct.
    for i in range(10):
        store.append_message("s1", "user" if i % 2 == 0 else "assistant", f"msg {i}")
    messages = store.get_messages("s1")
    assert [m["content"] for m in messages] == [f"msg {i}" for i in range(10)]
