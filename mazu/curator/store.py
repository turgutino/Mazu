import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS curator_runs (
    id             TEXT PRIMARY KEY,
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    model          TEXT NOT NULL,
    areas          TEXT NOT NULL,
    dry_run        INTEGER NOT NULL,
    status         TEXT NOT NULL,
    stop_reason    TEXT,
    total_cost_usd REAL
);

CREATE TABLE IF NOT EXISTS curator_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    curator_run_id  TEXT NOT NULL,
    area            TEXT NOT NULL,
    action          TEXT NOT NULL,
    target_type     TEXT,
    target_id       TEXT,
    rationale       TEXT NOT NULL,
    reversal_hint   TEXT,
    applied         INTEGER NOT NULL DEFAULT 1,
    outcome         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS curator_state (
    area         TEXT PRIMARY KEY,
    last_run_at  TEXT,
    last_run_id  TEXT
);

-- The lightweight "memory edges" store discussed during design -- two int columns
-- and a rationale, deliberately NOT a graph database and NOT inside memory.db
-- itself (keeps memory.db's schema untouched by this Phase-4-only concern).
CREATE TABLE IF NOT EXISTS memory_conflicts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at   TEXT NOT NULL,
    memory_id_a   INTEGER NOT NULL,
    memory_id_b   INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    confidence    REAL,
    rationale     TEXT NOT NULL,
    resolved_at   TEXT,
    resolution    TEXT
);

CREATE INDEX IF NOT EXISTS idx_curator_log_run ON curator_log(curator_run_id);
CREATE INDEX IF NOT EXISTS idx_curator_log_area ON curator_log(area, created_at);
CREATE INDEX IF NOT EXISTS idx_curator_runs_started ON curator_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_memory_conflicts_unresolved ON memory_conflicts(resolved_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_curator_run_id() -> str:
    return f"cur_{uuid.uuid4().hex[:12]}"


class CuratorStore:
    """Project-scoped (`.mazu/curator.db`), deliberately kept separate from
    memory.db/runs.db so Curator stays fully removable -- delete this one file and
    nothing else in Mazu notices or changes behavior. Records every Curator pass
    (curator_runs), every individual decision it made with its rationale and a
    reversal hint (curator_log -- the diary/self-report), and a per-area watermark
    (curator_state) so `mazu curator run` never blindly re-analyzes everything on
    every invocation.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def start_run(self, run_id: str, model: str, areas: list[str], dry_run: bool) -> None:
        self.conn.execute(
            "INSERT INTO curator_runs (id, started_at, model, areas, dry_run, status) "
            "VALUES (?, ?, ?, ?, ?, 'running')",
            (run_id, _now(), model, ",".join(areas), int(dry_run)),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, stop_reason: str | None, total_cost_usd: float) -> None:
        self.conn.execute(
            "UPDATE curator_runs SET ended_at = ?, status = ?, stop_reason = ?, "
            "total_cost_usd = ? WHERE id = ?",
            (_now(), status, stop_reason, total_cost_usd, run_id),
        )
        self.conn.commit()

    def log_entry(
        self,
        run_id: str,
        area: str,
        action: str,
        rationale: str,
        target_type: str | None = None,
        target_id: str | None = None,
        reversal_hint: str | None = None,
        applied: bool = True,
        outcome: str = "ok",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO curator_log (created_at, curator_run_id, area, action, target_type, "
            "target_id, rationale, reversal_hint, applied, outcome) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), run_id, area, action, target_type, target_id, rationale, reversal_hint, int(applied), outcome),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM curator_log WHERE curator_run_id = ? ORDER BY id", (run_id,)
        ).fetchall()

    def log_recent(self, area: str | None = None, since_days: int | None = None, limit: int = 50) -> list[sqlite3.Row]:
        conditions = []
        params: list = []
        if area is not None:
            conditions.append("area = ?")
            params.append(area)
        if since_days is not None:
            conditions.append("created_at >= datetime('now', ?)")
            params.append(f"-{since_days} days")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        return self.conn.execute(
            f"SELECT * FROM curator_log {where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()

    def last_run(self, run_id: str | None = None) -> sqlite3.Row | None:
        if run_id is not None:
            return self.conn.execute("SELECT * FROM curator_runs WHERE id = ?", (run_id,)).fetchone()
        return self.conn.execute(
            "SELECT * FROM curator_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

    def get_watermark(self, area: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM curator_state WHERE area = ?", (area,)).fetchone()

    def advance_watermark(self, area: str, run_id: str) -> None:
        self.conn.execute(
            "INSERT INTO curator_state (area, last_run_at, last_run_id) VALUES (?, ?, ?) "
            "ON CONFLICT(area) DO UPDATE SET last_run_at = excluded.last_run_at, "
            "last_run_id = excluded.last_run_id",
            (area, _now(), run_id),
        )
        self.conn.commit()

    def record_conflict(
        self, memory_id_a: int, memory_id_b: int, kind: str, rationale: str, confidence: float | None = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO memory_conflicts (detected_at, memory_id_a, memory_id_b, kind, "
            "confidence, rationale) VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), memory_id_a, memory_id_b, kind, confidence, rationale),
        )
        self.conn.commit()
        return cur.lastrowid

    def resolve_conflict(self, conflict_id: int, resolution: str) -> bool:
        cur = self.conn.execute(
            "UPDATE memory_conflicts SET resolved_at = ?, resolution = ? WHERE id = ?",
            (_now(), resolution, conflict_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def list_conflicts(self, unresolved_only: bool = True) -> list[sqlite3.Row]:
        if unresolved_only:
            return self.conn.execute(
                "SELECT * FROM memory_conflicts WHERE resolved_at IS NULL ORDER BY detected_at DESC"
            ).fetchall()
        return self.conn.execute("SELECT * FROM memory_conflicts ORDER BY detected_at DESC").fetchall()

    def close(self) -> None:
        self.conn.close()
