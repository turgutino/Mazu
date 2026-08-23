import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mazu.checkpoint.lock import ReentrantFileLock

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT NOT NULL,
    command            TEXT NOT NULL,
    session_id         TEXT,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    input_tokens       INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    estimated_cost_usd REAL
);

CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_log(created_at);
CREATE INDEX IF NOT EXISTS idx_usage_provider_model ON usage_log(provider, model);
"""

# Additive migration, same reasoning/pattern as mazu/runs/store.py's
# LINEAGE_COLUMNS/EXPLORE_OUTCOME_COLUMNS: `CREATE TABLE IF NOT EXISTS` above only
# creates these columns for a brand-new usage.db -- an existing one (every real
# user's ~/.mazu/usage.db) needs them added via ALTER TABLE, checked via
# PRAGMA table_info so it's a no-op once already applied. NULL by default (not 0):
# a NULL cache_read_tokens means "this provider/call never reported a cache figure"
# (Gemini, local models, or calls logged before this migration), distinct from a
# real, confirmed 0.
CACHE_TOKEN_COLUMNS = {"cache_read_tokens": "INTEGER", "cache_write_tokens": "INTEGER"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UsageStore:
    """Cross-project log of every model call Mazu has made, with an approximate USD
    cost per call (from llm/pricing.py's rough, documented-as-inexact table). Lives
    at ~/.mazu/usage.db by default -- global like global_memory.db, since spend is
    tied to the person/API keys, not any one project. A separate file from
    global_memory.db on purpose: this is telemetry (high write frequency, its own
    schema), not agent-facing knowledge, and keeping them apart avoids coupling
    MemoryStore to a concern it has nothing to do with.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # `mazu explore` runs multiple branches in parallel (each its own thread,
        # sometimes each its own worktree process), and every one constructs a
        # UsageStore against the same shared, global ~/.mazu/usage.db. On a brand
        # new db, two callers can both see the cache columns missing (read before
        # either has written) and both attempt `ALTER TABLE ADD COLUMN` -- the
        # second one loses with "duplicate column name". A cross-process file lock
        # (the same primitive mazu/checkpoint/lock.py already uses for the
        # analogous checkpoint-index race) serializes schema setup so only one
        # caller ever actually applies the DDL; every other caller's identical
        # CREATE/ALTER becomes a safe no-op (CREATE TABLE IF NOT EXISTS, and the
        # PRAGMA table_info check below) once it acquires the lock afterward.
        with ReentrantFileLock(db_path).acquire():
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(SCHEMA)
            self._migrate_cache_token_columns()
            self.conn.commit()

    def _migrate_cache_token_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(usage_log)")}
        for column, sql_type in CACHE_TOKEN_COLUMNS.items():
            if column not in existing:
                self.conn.execute(f"ALTER TABLE usage_log ADD COLUMN {column} {sql_type}")

    def log(
        self,
        command: str,
        session_id: str | None,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float | None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> None:
        # cache_read_tokens/cache_write_tokens default to None (not 0) so every
        # existing call site that never passes them keeps writing NULL, exactly as
        # before this migration -- the column existing is additive, this doesn't
        # change what a pre-existing caller records.
        self.conn.execute(
            "INSERT INTO usage_log "
            "(created_at, command, session_id, provider, model, input_tokens, "
            "output_tokens, estimated_cost_usd, cache_read_tokens, cache_write_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                command,
                session_id,
                provider,
                model,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
                cache_read_tokens,
                cache_write_tokens,
            ),
        )
        self.conn.commit()

    def summary(
        self,
        since_days: int | None = None,
        session_id: str | None = None,
        command: str | None = None,
    ) -> dict:
        """Aggregates spend grouped by (provider, model). `estimated_cost_usd` can be
        NULL for calls against a model with no pricing entry -- those still count
        toward `total_calls` but contribute 0 to `total_cost`, and are flagged via
        `has_unpriced_calls` so callers can show an honest caveat instead of silently
        underreporting spend.

        `session_id`, when given, scopes the summary to one run's own logged calls
        (usage_log.session_id is already populated on every row `run_autonomous`
        writes) -- this is what makes a per-run/per-branch estimated cost queryable
        for `mazu checkpoint compare-branches` without a separate cost-tracking table.

        `command`, when given, scopes to one `command` tag (e.g. "curator") -- the
        column already exists and is already populated ("run"/"chat"/"council"/...),
        this just adds the missing filter so a caller can isolate one kind of spend
        (e.g. `mazu usage --command curator`) from the rest of `mazu usage`.
        """
        conditions = []
        params: list = []
        if since_days is not None:
            conditions.append("created_at >= datetime('now', ?)")
            params.append(f"-{since_days} days")
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if command is not None:
            conditions.append("command = ?")
            params.append(command)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        rows = self.conn.execute(
            f"SELECT provider, model, COUNT(*) AS calls, "
            f"SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            f"SUM(cache_read_tokens) AS cache_read_tokens, "
            f"SUM(cache_write_tokens) AS cache_write_tokens, "
            f"SUM(estimated_cost_usd) AS cost, "
            f"SUM(CASE WHEN estimated_cost_usd IS NULL THEN 1 ELSE 0 END) AS unpriced_calls "
            f"FROM usage_log {where} GROUP BY provider, model ORDER BY cost DESC",
            params,
        ).fetchall()

        by_model = [dict(r) for r in rows]
        return {
            "by_model": by_model,
            "total_cost": sum(r["cost"] or 0.0 for r in by_model),
            "total_calls": sum(r["calls"] for r in by_model),
            "has_unpriced_calls": any(r["unpriced_calls"] for r in by_model),
            # NULL (not 0) whenever every row in scope predates this migration or
            # came from a provider that never reports caching -- distinguishes "no
            # cache data available" from "confirmed zero cache usage".
            "total_cache_read_tokens": sum(r["cache_read_tokens"] or 0 for r in by_model),
            "total_cache_write_tokens": sum(r["cache_write_tokens"] or 0 for r in by_model),
        }

    def close(self) -> None:
        self.conn.close()
