"""Thin data-loading layer for the Terminal UI (mazu/ui/app.py). Deliberately has no
Textual import anywhere in this module -- every function here just wraps an existing
store's already-structured return values (list[dict]/sqlite3.Row) into small
dataclasses the UI renders, so the data layer is testable on its own, the same way
the rest of Mazu's business logic already is, without spinning up a terminal app.
"""

from dataclasses import dataclass

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.memory.store import MemoryStore


@dataclass
class CheckpointRow:
    id: str
    created_at: str
    trigger: str
    summary: str
    files_changed: list[str]
    branch: str
    parent_checkpoint_id: str | None


def load_checkpoints(checkpoint_manager: CheckpointManager) -> list[CheckpointRow]:
    """Newest first -- timeline_entries() itself is oldest-first (the natural order
    for "what changed since the previous one"), reversed here since the UI's table
    should show the most recent, most likely-relevant checkpoint at the top.
    """
    entries = checkpoint_manager.timeline_entries()
    rows = [
        CheckpointRow(
            id=e["id"],
            created_at=e["created_at"],
            trigger=e["trigger"],
            summary=e["summary"],
            files_changed=e["files_changed"],
            # Both optional/additive on the underlying entry -- checkpoints recorded
            # before branching existed simply lack these keys.
            branch=e.get("branch") or "(unknown)",
            parent_checkpoint_id=e.get("parent_checkpoint_id"),
        )
        for e in entries
    ]
    return list(reversed(rows))


@dataclass
class MemoryRow:
    id: int
    category: str
    title: str
    body: str
    pinned: bool
    tags: str


def load_memories(memory_store: MemoryStore) -> list[MemoryRow]:
    rows = memory_store.search(limit=500)
    return [
        MemoryRow(
            id=r["id"],
            category=r["category"],
            title=r["title"],
            body=r["body"],
            pinned=bool(r["pinned"]),
            tags=r["tags"] or "",
        )
        for r in rows
    ]


@dataclass
class SessionRow:
    session_id: str
    command: str
    action_count: int
    error_count: int
    started_at: str
    last_at: str


def load_sessions(action_log_store: ActionLogStore, limit: int = 100) -> list[SessionRow]:
    rows = action_log_store.list_sessions(limit=limit)
    return [
        SessionRow(
            session_id=r["session_id"],
            command=r["command"],
            action_count=r["action_count"],
            error_count=r["error_count"],
            started_at=r["started_at"],
            last_at=r["last_at"],
        )
        for r in rows
    ]


@dataclass
class ActionRow:
    created_at: str
    tool_name: str
    outcome: str
    output_summary: str
    changed_file: str | None


def load_actions(action_log_store: ActionLogStore, session_id: str) -> list[ActionRow]:
    rows = action_log_store.session_actions(session_id)
    return [
        ActionRow(
            created_at=r["created_at"],
            tool_name=r["tool_name"],
            outcome=r["outcome"],
            output_summary=r["output_summary"],
            changed_file=r["changed_file"],
        )
        for r in rows
    ]
