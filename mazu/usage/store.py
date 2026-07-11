import sqlite3
from datetime import datetime, timezone
from pathlib import Path

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
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def log(
        self,
        command: str,
        session_id: str | None,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float | None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO usage_log "
            "(created_at, command, session_id, provider, model, input_tokens, "
            "output_tokens, estimated_cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                command,
                session_id,
                provider,
                model,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
            ),
        )
        self.conn.commit()

    def summary(self, since_days: int | None = None) -> dict:
        """Aggregates spend grouped by (provider, model). `estimated_cost_usd` can be
        NULL for calls against a model with no pricing entry -- those still count
        toward `total_calls` but contribute 0 to `total_cost`, and are flagged via
        `has_unpriced_calls` so callers can show an honest caveat instead of silently
        underreporting spend.
        """
        where = ""
        params: list = []
        if since_days is not None:
            where = "WHERE created_at >= datetime('now', ?)"
            params.append(f"-{since_days} days")

        rows = self.conn.execute(
            f"SELECT provider, model, COUNT(*) AS calls, "
            f"SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
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
        }

    def close(self) -> None:
        self.conn.close()
