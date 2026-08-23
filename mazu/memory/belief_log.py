"""Append-only log of belief-maintenance shadow-mode decisions -- Phase 3's
evidence-collection mechanism. Deliberately a separate sqlite file from
memory.db, same reasoning as curator.db (mazu/curator/store.py): this is
diagnostic/audit data about the classifier's behavior, not memory data itself,
and keeping it separate means memory.db's own schema/tests are never coupled to
this still-experimental, shadow-only mechanism. Nothing in this module ever
writes to or reads from the memories table.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_decisions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT NOT NULL,
    session_id         TEXT,
    category           TEXT NOT NULL,
    new_title          TEXT NOT NULL,
    new_body           TEXT NOT NULL,
    related_memory_id  INTEGER,
    related_title      TEXT,
    similarity_score   REAL,
    relationship       TEXT,
    confidence         TEXT,
    proposed_action    TEXT NOT NULL,
    would_supersede    INTEGER NOT NULL DEFAULT 0,
    classifier_failed  INTEGER NOT NULL DEFAULT 0,
    reasoning          TEXT
);

CREATE INDEX IF NOT EXISTS idx_shadow_decisions_created ON shadow_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_decisions_session ON shadow_decisions(session_id);

-- One row per real, explicit `remember` tool call (mazu/tools/memory_tools.py) --
-- an agent-initiated memory write/supersede, as opposed to shadow_decisions'
-- auto-extraction-time gray-zone classifications. No classifier is ever involved
-- here: `asserted_relationship` is only ever "replacement" when the agent itself
-- supplied supersedes_id (the agent told us the relationship by acting on it),
-- and NULL for a plain insert -- never guessed. `nearest_active_*` is a purely
-- local (zero-API-cost) note of the closest existing same-category memory the
-- agent chose NOT to touch on a plain insert, so a human reviewer can later spot
-- "the agent probably should have superseded/refined this but didn't."
CREATE TABLE IF NOT EXISTS explicit_memory_actions (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at                TEXT NOT NULL,
    session_id                TEXT,
    project                   TEXT,
    action_type               TEXT NOT NULL,  -- 'insert' or 'supersede'
    category                  TEXT NOT NULL,
    new_memory_id             INTEGER,
    new_title                 TEXT NOT NULL,
    new_body                  TEXT NOT NULL,
    old_memory_id             INTEGER,
    old_title                 TEXT,
    old_body                  TEXT,
    similarity_score          REAL,
    asserted_relationship     TEXT,
    nearest_active_memory_id  INTEGER,
    nearest_active_title      TEXT,
    nearest_active_similarity REAL,
    tool_result               TEXT NOT NULL,
    tool_success              INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_explicit_actions_created ON explicit_memory_actions(created_at);
CREATE INDEX IF NOT EXISTS idx_explicit_actions_session ON explicit_memory_actions(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BeliefShadowLogStore:
    """One row per (new extracted memory, related existing memory) pair that
    landed in the gray zone and was classified -- this is the raw evidence
    reviewed before any future phase is allowed to let the classifier actually
    mutate memory. `would_supersede` and `proposed_action` record what WOULD
    have happened; nothing in this class or its caller applies that action.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def log_decision(
        self,
        session_id: str | None,
        category: str,
        new_title: str,
        new_body: str,
        related_memory_id: int | None,
        related_title: str | None,
        similarity_score: float | None,
        relationship: str | None,
        confidence: str | None,
        proposed_action: str,
        would_supersede: bool,
        classifier_failed: bool,
        reasoning: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO shadow_decisions "
            "(created_at, session_id, category, new_title, new_body, related_memory_id, "
            "related_title, similarity_score, relationship, confidence, proposed_action, "
            "would_supersede, classifier_failed, reasoning) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(), session_id, category, new_title, new_body, related_memory_id,
                related_title, similarity_score, relationship, confidence, proposed_action,
                int(would_supersede), int(classifier_failed), reasoning,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def all_decisions(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM shadow_decisions ORDER BY created_at ASC"
        ).fetchall()

    def summary_by_relationship(self) -> dict[str, int]:
        return {
            row["relationship"]: row["c"]
            for row in self.conn.execute(
                "SELECT relationship, COUNT(*) c FROM shadow_decisions GROUP BY relationship"
            )
        }

    def stats(self) -> dict:
        """Read-only aggregate for `mazu memory-belief stats` -- every figure here
        is a real COUNT/AVG over shadow_decisions, never estimated or fabricated.
        Callers must treat a small `total` as "not enough evidence yet", not
        extrapolate from it (see mazu.cli's MIN_DECISIONS_FOR_STATS caveat).
        """
        total = self.conn.execute("SELECT COUNT(*) c FROM shadow_decisions").fetchone()["c"]

        def _group_counts(column: str) -> dict[str, int]:
            return {
                row[column]: row["c"]
                for row in self.conn.execute(
                    f"SELECT {column}, COUNT(*) c FROM shadow_decisions GROUP BY {column}"
                )
            }

        avg_similarity = self.conn.execute(
            "SELECT AVG(similarity_score) a FROM shadow_decisions WHERE similarity_score IS NOT NULL"
        ).fetchone()["a"]
        classifier_failed_count = self.conn.execute(
            "SELECT COUNT(*) c FROM shadow_decisions WHERE classifier_failed = 1"
        ).fetchone()["c"]
        would_supersede_count = self.conn.execute(
            "SELECT COUNT(*) c FROM shadow_decisions WHERE would_supersede = 1"
        ).fetchone()["c"]
        flagged_for_review_count = self.conn.execute(
            "SELECT COUNT(*) c FROM shadow_decisions WHERE proposed_action = 'flag_for_review'"
        ).fetchone()["c"]

        return {
            "total": total,
            "by_relationship": _group_counts("relationship"),
            "by_confidence": _group_counts("confidence"),
            "by_proposed_action": _group_counts("proposed_action"),
            "classifier_failed_count": classifier_failed_count,
            "avg_similarity_score": avg_similarity,
            "would_supersede_count": would_supersede_count,
            "flagged_for_review_count": flagged_for_review_count,
        }

    def recent(
        self,
        limit: int = 20,
        relationship: str | None = None,
        confidence: str | None = None,
    ) -> list[sqlite3.Row]:
        """Read-only. Most-recent-first, optionally filtered -- used by
        `mazu memory-belief review`. Never mutates shadow_decisions or memories.
        """
        conditions = []
        params: list = []
        if relationship is not None:
            conditions.append("relationship = ?")
            params.append(relationship)
        if confidence is not None:
            conditions.append("confidence = ?")
            params.append(confidence)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        return self.conn.execute(
            f"SELECT * FROM shadow_decisions {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()

    def log_explicit_action(
        self,
        session_id: str | None,
        project: str | None,
        action_type: str,
        category: str,
        new_memory_id: int | None,
        new_title: str,
        new_body: str,
        old_memory_id: int | None = None,
        old_title: str | None = None,
        old_body: str | None = None,
        similarity_score: float | None = None,
        asserted_relationship: str | None = None,
        nearest_active_memory_id: int | None = None,
        nearest_active_title: str | None = None,
        nearest_active_similarity: float | None = None,
        tool_result: str = "",
        tool_success: bool = True,
    ) -> int:
        """Records one real, explicit `remember` tool call -- purely observational,
        called AFTER the real action has already happened (see
        mazu.tools.memory_tools.make_memory_tools). Never involves an LLM call.
        """
        cur = self.conn.execute(
            "INSERT INTO explicit_memory_actions "
            "(created_at, session_id, project, action_type, category, new_memory_id, new_title, "
            "new_body, old_memory_id, old_title, old_body, similarity_score, asserted_relationship, "
            "nearest_active_memory_id, nearest_active_title, nearest_active_similarity, "
            "tool_result, tool_success) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(), session_id, project, action_type, category, new_memory_id, new_title, new_body,
                old_memory_id, old_title, old_body, similarity_score, asserted_relationship,
                nearest_active_memory_id, nearest_active_title, nearest_active_similarity,
                tool_result, int(tool_success),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def all_explicit_actions(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM explicit_memory_actions ORDER BY created_at ASC"
        ).fetchall()

    def explicit_actions_stats(self) -> dict:
        """Read-only aggregate, same "never fabricate, small-N is small-N" spirit
        as stats() above."""
        total = self.conn.execute("SELECT COUNT(*) c FROM explicit_memory_actions").fetchone()["c"]
        by_action_type = {
            row["action_type"]: row["c"]
            for row in self.conn.execute(
                "SELECT action_type, COUNT(*) c FROM explicit_memory_actions GROUP BY action_type"
            )
        }
        supersede_count = by_action_type.get("supersede", 0)
        insert_with_nearby_unactioned = self.conn.execute(
            "SELECT COUNT(*) c FROM explicit_memory_actions "
            "WHERE action_type = 'insert' AND nearest_active_memory_id IS NOT NULL"
        ).fetchone()["c"]
        failed_count = self.conn.execute(
            "SELECT COUNT(*) c FROM explicit_memory_actions WHERE tool_success = 0"
        ).fetchone()["c"]
        return {
            "total": total,
            "by_action_type": by_action_type,
            "supersede_count": supersede_count,
            "insert_with_nearby_unactioned_count": insert_with_nearby_unactioned,
            "failed_count": failed_count,
        }

    def recent_explicit_actions(
        self,
        limit: int = 20,
        action_type: str | None = None,
    ) -> list[sqlite3.Row]:
        conditions = []
        params: list = []
        if action_type is not None:
            conditions.append("action_type = ?")
            params.append(action_type)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        return self.conn.execute(
            f"SELECT * FROM explicit_memory_actions {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()

    def close(self) -> None:
        self.conn.close()
