import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    model      TEXT,
    title      TEXT
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at);
"""

TITLE_MAX_CHARS = 80


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_title(content) -> str | None:
    # The first user message is always plain text in mazu's chat loop (tool-result
    # continuation messages, which carry a list of blocks instead, are never the
    # first message of a session) -- but this stays defensive rather than assuming
    # that shape, since a resumed/forked session could in principle start elsewhere.
    if not isinstance(content, str):
        return None
    text = content.strip().replace("\n", " ")
    if len(text) <= TITLE_MAX_CHARS:
        return text
    return text[:TITLE_MAX_CHARS] + "..."


class ChatStore:
    """Project-scoped (`.mazu/chat_history.db`, mirroring memory.db/action_log.db)
    durable transcript of every `mazu chat` / mazu-web chat conversation.

    Distinct from CheckpointManager's `conversation.json` snapshots: those are
    heavy (git commit + memory/skills backup per snapshot) and only ever taken on
    the manual `/checkpoint` command, so a chat that's never checkpointed has no
    recoverable history at all -- exactly the bug this closes. append_message() is
    called at every single message boundary (each user message, each assistant
    turn, each tool-result round), not just at turn-end, so a mid-turn crash loses
    at most the one in-flight LLM call, never anything already appended to the
    live `messages` list.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def append_message(self, session_id: str, role: str, content, model: str | None = None) -> None:
        now = _now()
        content_json = json.dumps(content, default=str)
        seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        self.conn.execute(
            "INSERT INTO chat_messages (session_id, seq, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, seq, role, content_json, now),
        )
        exists = self.conn.execute(
            "SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if exists is None:
            self.conn.execute(
                "INSERT INTO chat_sessions (session_id, started_at, updated_at, model, title) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, now, now, model, _derive_title(content) if role == "user" else None),
            )
        else:
            self.conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id)
            )
        self.conn.commit()

    def list_sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        """Most recently active first, with a live message_count -- not stored
        denormalized on chat_sessions, computed via subquery so it can never drift
        from the actual chat_messages rows.
        """
        return self.conn.execute(
            "SELECT s.session_id, s.started_at, s.updated_at, s.model, s.title, "
            "(SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.session_id) AS message_count "
            "FROM chat_sessions s ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

    def get_messages(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
        return [{"role": r["role"], "content": json.loads(r["content"])} for r in rows]

    def latest_session_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT session_id FROM chat_sessions ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return row["session_id"] if row is not None else None

    def delete_session(self, session_id: str) -> bool:
        """Permanently deletes a session and every message in it. Returns False
        (a no-op, not an error) if the session didn't exist -- same contract as
        MemoryStore.forget()/unset_config_value().
        """
        cur = self.conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self.conn.close()
