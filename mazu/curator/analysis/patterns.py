from mazu.action_log.store import ActionLogStore


def find_common_tool_sequences(
    action_log_store: ActionLogStore, min_count: int = 3, session_limit: int = 200
) -> list[dict]:
    """Counts consecutive (tool_a -> tool_b) pairs within the same session -- a
    crude but real signal for "this project keeps doing X then Y", which Curator
    can use to suggest (via write_skill, in the skills area) a single skill that
    does both steps at once. Read-only; no schema change needed since
    ActionLogStore.session_actions() already returns rows in chronological order.
    """
    sessions = action_log_store.list_sessions(limit=session_limit)
    pair_counts: dict[tuple[str, str], int] = {}
    for session in sessions:
        rows = action_log_store.session_actions(session["session_id"])
        for a, b in zip(rows, rows[1:]):
            pair = (a["tool_name"], b["tool_name"])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return [
        {"sequence": list(pair), "count": count}
        for pair, count in sorted(pair_counts.items(), key=lambda kv: -kv[1])
        if count >= min_count
    ]
