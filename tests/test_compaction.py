"""Tests for context compaction (Faza 2). The correctness bar here is strict: the
compacted message list must still be a *valid* API request -- strictly alternating
user/assistant roles, and never a tool_use block separated from its paired
tool_result. All summarization is mocked (no API calls); these tests are entirely
about the message-list surgery being correct, not about summary quality.
"""

import pytest

from mazu.agent.compaction import (
    compact_if_needed,
    compact_messages,
    estimate_tokens,
    force_compact,
    needs_compaction,
    _round_boundaries,
)


def _user(text) -> dict:
    return {"role": "user", "content": text}


def _assistant_text(text) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _assistant_tool_use(tool_id, name="read_file", input_=None) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": input_ or {}}],
    }


def _tool_result(tool_id, content="ok") -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}]}


def _assert_strictly_alternating(messages: list[dict]) -> None:
    assert messages, "expected a non-empty message list"
    assert messages[0]["role"] == "user"
    for prev, cur in zip(messages, messages[1:]):
        assert prev["role"] != cur["role"], f"consecutive same-role messages: {prev['role']}"


def _assert_no_orphaned_tool_use(messages: list[dict]) -> None:
    """Every assistant tool_use block must be immediately followed by a user message
    containing a matching tool_result -- otherwise the request is invalid.
    """
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        tool_use_ids = [b["id"] for b in msg["content"] if b.get("type") == "tool_use"]
        if not tool_use_ids:
            continue
        assert i + 1 < len(messages), "tool_use with no following message at all"
        next_msg = messages[i + 1]
        assert next_msg["role"] == "user"
        result_ids = [b["tool_use_id"] for b in next_msg["content"] if b.get("type") == "tool_result"]
        assert set(tool_use_ids) <= set(result_ids)


# ---------------------------------------------------------------------------
# estimate_tokens / needs_compaction
# ---------------------------------------------------------------------------


def test_estimate_tokens_roughly_chars_over_four():
    messages = [_user("a" * 400)]
    assert estimate_tokens(messages) == 100


def test_needs_compaction_below_threshold_false():
    messages = [_user("short")]
    assert needs_compaction(messages, trigger_tokens=1000) is False


def test_needs_compaction_above_threshold_true():
    messages = [_user("x" * 40_000)]
    assert needs_compaction(messages, trigger_tokens=1000) is True


# ---------------------------------------------------------------------------
# _round_boundaries
# ---------------------------------------------------------------------------


def test_round_boundaries_simple_alternating_conversation():
    messages = [_user("hi"), _assistant_text("hello"), _user("do X"), _assistant_text("done")]
    # No tool_use anywhere -- every position is safe, including start and end.
    assert _round_boundaries(messages) == [0, 1, 2, 3, 4]


def test_round_boundaries_excludes_gap_inside_tool_use_pair():
    messages = [
        _user("do X"),
        _assistant_tool_use("t1"),
        _tool_result("t1"),
        _assistant_text("done"),
    ]
    boundaries = _round_boundaries(messages)
    # Index 2 (between the tool_use at index 1 and its result at index 2) must be
    # excluded -- cutting there would separate the tool_use from its tool_result.
    assert 2 not in boundaries
    assert 0 in boundaries
    assert 1 in boundaries  # cutting right before the tool_use itself is fine
    assert 3 in boundaries  # right after the pair is resolved
    assert 4 in boundaries


# ---------------------------------------------------------------------------
# compact_messages
# ---------------------------------------------------------------------------


def test_compact_messages_noop_when_short():
    messages = [_user("hi"), _assistant_text("hello")]
    result = compact_messages(messages, summarize_fn=lambda mm: "SUMMARY", keep_recent=10)
    assert result is messages  # unchanged, same object


def test_compact_messages_produces_alternating_result():
    messages = []
    for i in range(20):
        messages.append(_user(f"task {i}"))
        messages.append(_assistant_text(f"response {i}"))

    result = compact_messages(messages, summarize_fn=lambda mm: "SUMMARY", keep_recent=6)

    _assert_strictly_alternating(result)
    assert len(result) < len(messages)
    assert result[0]["content"].startswith("## Summary of earlier conversation")
    assert result[1]["role"] == "assistant"


def test_compact_messages_keeps_recent_tail_verbatim():
    messages = []
    for i in range(20):
        messages.append(_user(f"task {i}"))
        messages.append(_assistant_text(f"response {i}"))

    result = compact_messages(messages, summarize_fn=lambda mm: "SUMMARY", keep_recent=6)

    # The last 6 original messages must appear, byte-for-byte, unchanged at the tail.
    assert result[-6:] == messages[-6:]


def test_compact_messages_never_splits_tool_use_pair():
    messages = [_user("start")]
    for i in range(15):
        messages.append(_assistant_tool_use(f"t{i}"))
        messages.append(_tool_result(f"t{i}"))
    messages.append(_assistant_text("final answer"))

    result = compact_messages(messages, summarize_fn=lambda mm: "SUMMARY", keep_recent=6)

    _assert_strictly_alternating(result)
    _assert_no_orphaned_tool_use(result)
    assert len(result) < len(messages)


def test_compact_messages_no_valid_cut_point_is_noop():
    # A single giant unresolved round (assistant tool_use with no result yet) at the
    # very start -- there's no safe place to cut that leaves a real middle chunk.
    messages = [_user("start"), _assistant_tool_use("t1")]
    result = compact_messages(messages, summarize_fn=lambda mm: "SUMMARY", keep_recent=1)
    assert result is messages


def test_compact_messages_summarize_fn_receives_the_head_only():
    messages = []
    for i in range(20):
        messages.append(_user(f"task {i}"))
        messages.append(_assistant_text(f"response {i}"))

    captured = {}

    def _capture(mm):
        captured["messages"] = mm
        return "SUMMARY"

    result = compact_messages(messages, summarize_fn=_capture, keep_recent=6)

    # Whatever was summarized should be exactly the portion that's no longer present
    # verbatim in the result (i.e. not part of the kept tail).
    assert captured["messages"] == messages[: len(messages) - 6]


# ---------------------------------------------------------------------------
# compact_if_needed / force_compact (the two entry points actually wired into
# the agent loops)
# ---------------------------------------------------------------------------


def test_compact_if_needed_noop_below_trigger():
    messages = [_user("hi"), _assistant_text("hello")]
    result, did_compact = compact_if_needed(messages, model=None, trigger_tokens=1_000_000)
    assert did_compact is False
    assert result is messages


def test_compact_if_needed_compacts_above_trigger(monkeypatch):
    messages = []
    for i in range(20):
        messages.append(_user(f"task {i}" * 50))
        messages.append(_assistant_text(f"response {i}" * 50))

    monkeypatch.setattr(
        "mazu.agent.compaction.summarize_for_compaction", lambda mm, model: "SUMMARY"
    )

    result, did_compact = compact_if_needed(messages, model=None, trigger_tokens=10, keep_recent=6)

    assert did_compact is True
    _assert_strictly_alternating(result)
    assert len(result) < len(messages)


def test_force_compact_uses_aggressive_keep_recent(monkeypatch):
    messages = []
    for i in range(20):
        messages.append(_user(f"task {i}"))
        messages.append(_assistant_text(f"response {i}"))

    monkeypatch.setattr(
        "mazu.agent.compaction.summarize_for_compaction", lambda mm, model: "SUMMARY"
    )

    result = force_compact(messages, model=None)

    # AGGRESSIVE_KEEP_RECENT (4) < DEFAULT_KEEP_RECENT (10) -- fewer messages survive.
    assert len(result) - 2 == 4  # minus the 2 synthetic summary/ack messages
    _assert_strictly_alternating(result)
