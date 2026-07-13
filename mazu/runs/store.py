import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                  TEXT PRIMARY KEY,
    task                TEXT NOT NULL,
    model               TEXT,
    max_steps           INTEGER NOT NULL,
    checkpoint_every    INTEGER NOT NULL,
    allow_shell         INTEGER NOT NULL,
    shell_allowlist     TEXT,
    max_cost            REAL,
    dry_run             INTEGER NOT NULL,
    status              TEXT NOT NULL,
    stop_reason         TEXT,
    started_at          TEXT NOT NULL,
    ended_at            TEXT,
    last_step           INTEGER NOT NULL DEFAULT 0,
    last_checkpoint_id  TEXT,
    checkpoints_created INTEGER NOT NULL DEFAULT 0,
    memories_saved      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Project-scoped (`.mazu/runs.db`, mirroring memory.db/action_log.db) record of
    every `mazu run` invocation: its id (== session_id, the same id already used by
    ActionLogStore/UsageStore/MemoryStore.sessions -- this formalizes it as a
    first-class "run", not a new id scheme), the config it started with, its live
    progress, and how it ended. This is what `mazu run --resume <run_id>` reads to
    recover a run's original settings, and what the end-of-run report is built from.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def start(
        self,
        run_id: str,
        task: str,
        model: str | None,
        max_steps: int,
        checkpoint_every: int,
        allow_shell: bool,
        shell_allowlist: list[str] | None,
        max_cost: float | None,
        dry_run: bool,
    ) -> None:
        self.conn.execute(
            "INSERT INTO runs "
            "(id, task, model, max_steps, checkpoint_every, allow_shell, shell_allowlist, "
            "max_cost, dry_run, status, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)",
            (
                run_id,
                task,
                model,
                max_steps,
                checkpoint_every,
                int(allow_shell),
                ",".join(shell_allowlist) if shell_allowlist else None,
                max_cost,
                int(dry_run),
                _now(),
            ),
        )
        self.conn.commit()

    def update_progress(self, run_id: str, step: int, checkpoint_id: str | None = None) -> None:
        if checkpoint_id is not None:
            self.conn.execute(
                "UPDATE runs SET last_step = ?, last_checkpoint_id = ?, "
                "checkpoints_created = checkpoints_created + 1 WHERE id = ?",
                (step, checkpoint_id, run_id),
            )
        else:
            self.conn.execute("UPDATE runs SET last_step = ? WHERE id = ?", (step, run_id))
        self.conn.commit()

    def finish(
        self, run_id: str, status: str, stop_reason: str, memories_saved: int = 0
    ) -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, stop_reason = ?, ended_at = ?, "
            "memories_saved = memories_saved + ? WHERE id = ?",
            (status, stop_reason, _now(), memories_saved, run_id),
        )
        self.conn.commit()

    def get(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def list_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def close(self) -> None:
        self.conn.close()
