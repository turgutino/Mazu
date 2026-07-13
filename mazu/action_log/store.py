import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    command        TEXT NOT NULL,
    tool_name      TEXT NOT NULL,
    tool_input     TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    output_summary TEXT NOT NULL,
    changed_file   TEXT
);

CREATE INDEX IF NOT EXISTS idx_actions_session ON actions(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_actions_created ON actions(created_at);
"""

# Kept well under SQLite's practical row-size comfort zone -- a tool call's input
# (e.g. write_file's full file content) or a shell command's stdout can otherwise
# balloon the log far beyond what "what did the agent do" inspection needs.
TOOL_INPUT_MAX_CHARS = 500
OUTPUT_SUMMARY_MAX_CHARS = 300

# Tools whose input includes a `path` field naming the file they wrote/edited --
# used to populate the `changed_file` column. Anything else (read-only tools,
# run_shell) leaves it NULL; a shell command can touch arbitrary files with no
# single structured path to record.
_FILE_WRITING_TOOLS = {"write_file", "edit_file"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(truncated, {len(text)} chars total)"


def _serialize_input(tool_input: dict) -> str:
    try:
        text = json.dumps(tool_input, default=str)
    except (TypeError, ValueError):
        text = str(tool_input)
    return _truncate(text, TOOL_INPUT_MAX_CHARS)


class ActionLogStore:
    """Project-scoped (`.mazu/action_log.db`, mirroring memory.db) persistent record of
    every tool call an agent session makes: what tool, what input, what happened, and
    what file (if any) it touched. This is the audit trail behind `mazu log` / `mazu log
    show <session_id>` -- distinct from UsageStore (global, cost/token telemetry) and
    from CheckpointManager (git+memory snapshots) even though all three key off the same
    session_id.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def log(
        self,
        session_id: str,
        command: str,
        tool_name: str,
        tool_input: str,
        outcome: str,
        output_summary: str,
        changed_file: str | None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO actions "
            "(created_at, session_id, command, tool_name, tool_input, outcome, "
            "output_summary, changed_file) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), session_id, command, tool_name, tool_input, outcome, output_summary, changed_file),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        """One row per session, most recently active first: session_id, the command
        that ran it (chat/run/council), how many actions it logged, how many of those
        weren't a clean "ok", and its first/last action timestamps.
        """
        return self.conn.execute(
            "SELECT session_id, command, "
            "MIN(created_at) AS started_at, MAX(created_at) AS last_at, "
            "COUNT(*) AS action_count, "
            "SUM(CASE WHEN outcome != 'ok' THEN 1 ELSE 0 END) AS error_count "
            "FROM actions GROUP BY session_id ORDER BY MAX(created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def session_actions(self, session_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM actions WHERE session_id = ? ORDER BY created_at ASC, id ASC",
            (session_id,),
        ).fetchall()

    def close(self) -> None:
        self.conn.close()


def record_action(
    store: "ActionLogStore | None",
    session_id: str,
    command: str,
    tool_name: str,
    tool_input: dict,
    outcome: str,
    output_summary: str,
) -> None:
    """Shared entry point for the three agent loops (chat/run/council) to record a
    single tool call -- centralizes input serialization/truncation and changed-file
    resolution so that logic isn't duplicated three times. `outcome` is one of "ok",
    "error", "blocked" (denylisted shell command), "declined" (user said no to a
    destructive tool), or "unknown_tool". A None `store` is a no-op, so call sites
    don't need to guard every call themselves.
    """
    if store is None:
        return
    changed_file = tool_input.get("path") if tool_name in _FILE_WRITING_TOOLS else None
    store.log(
        session_id=session_id,
        command=command,
        tool_name=tool_name,
        tool_input=_serialize_input(tool_input),
        outcome=outcome,
        output_summary=_truncate(output_summary, OUTPUT_SUMMARY_MAX_CHARS),
        changed_file=changed_file,
    )
