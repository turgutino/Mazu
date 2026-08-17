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

# Additive lineage columns for forked runs (mazu run --from-checkpoint), added after
# the original schema shipped. Nullable, default NULL for every pre-existing row and
# every ordinary (non-forked) run. Not part of SCHEMA above because CREATE TABLE IF
# NOT EXISTS never alters an existing table -- an already-created runs.db needs an
# explicit ALTER TABLE migration instead (see __init__).
LINEAGE_COLUMNS = ["origin_checkpoint_id", "parent_run_id", "branch_name"]

# Same additive-migration reasoning as LINEAGE_COLUMNS, added later for `mazu
# explore` (mazu/agent/explore.py) to record each branch's real outcome instead of
# discarding it once the report is printed. explore_group_id ties every branch of
# one `mazu explore` call together (NULL for an ordinary, non-explore run);
# test_passed is a nullable INTEGER tri-state (1/0/NULL for passed/failed/no
# --test-command given), mirroring the bool | None already used throughout
# explore.py. Deliberately does NOT include cost -- that stays exclusively in
# UsageStore, joined by session_id, so there is never a second, driftable copy of
# spend data (see _run_one_branch's own existing re-join pattern).
EXPLORE_OUTCOME_COLUMNS = {"explore_group_id": "TEXT", "test_passed": "INTEGER"}


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
        self._migrate_lineage_columns()
        self._migrate_explore_outcome_columns()
        self.conn.commit()

    def _migrate_lineage_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(runs)")}
        for column in LINEAGE_COLUMNS:
            if column not in existing:
                self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT")

    def _migrate_explore_outcome_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(runs)")}
        for column, sql_type in EXPLORE_OUTCOME_COLUMNS.items():
            if column not in existing:
                self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {sql_type}")

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
        origin_checkpoint_id: str | None = None,
        parent_run_id: str | None = None,
        branch_name: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO runs "
            "(id, task, model, max_steps, checkpoint_every, allow_shell, shell_allowlist, "
            "max_cost, dry_run, status, started_at, origin_checkpoint_id, parent_run_id, branch_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)",
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
                origin_checkpoint_id,
                parent_run_id,
                branch_name,
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

    def set_explore_outcome(
        self, run_id: str, explore_group_id: str, test_passed: bool | None
    ) -> None:
        """Records that `run_id` was one branch of a `mazu explore` comparison
        (explore_group_id) and whether its --test-command passed, if one was given.
        A separate method from finish() -- finish() is generic to every run
        (mazu run included), these two fields are explore-specific.
        """
        self.conn.execute(
            "UPDATE runs SET explore_group_id = ?, test_passed = ? WHERE id = ?",
            (explore_group_id, None if test_passed is None else int(test_passed), run_id),
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
