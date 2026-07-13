import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mazu.memory.embeddings import serialize_embedding

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")

# Below this, two memories are treated as unrelated rather than near-duplicates —
# tuned to catch obvious rephrasings of the same fact without flagging genuinely
# different facts that just happen to share a few common words. Titles alone are
# usually too short/sparse for word-overlap to be meaningful, so this compares
# title+body combined when body is available.
FUZZY_DUPLICATE_THRESHOLD = 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _memory_similarity(title_a: str, body_a: str, title_b: str, body_b: str) -> float:
    """Jaccard word-overlap similarity over title+body combined — deliberately
    simple (not embeddings/semantic), just enough to catch near-identical
    rephrasings that exact-title-match dedup misses. Real paraphrases with little
    shared vocabulary still won't be caught; that needs semantic search, which is
    intentionally out of scope for this MVP-level check.
    """
    wa = _word_set(f"{title_a} {body_a}")
    wb = _word_set(f"{title_b} {body_b}")
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_PATH.read_text())
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Additive-only migration for DBs created before a given column existed.
        CREATE TABLE IF NOT EXISTS (above) only applies the current schema to brand
        new databases -- an existing one (like anyone's real .mazu/memory.db or
        ~/.mazu/global_memory.db from before this column existed) needs an explicit
        ALTER TABLE, or every subsequent INSERT/SELECT referencing `embedding` would
        fail against it.
        """
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(memories)")}
        if "embedding" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
            self.conn.commit()

    def add(
        self,
        category: str,
        title: str,
        body: str,
        tags: str = "",
        source: str = "explicit",
        session_id: str | None = None,
        pinned: bool = False,
        embedding: list[float] | None = None,
    ) -> int:
        now = _now()
        embedding_json = serialize_embedding(embedding) if embedding is not None else None
        cur = self.conn.execute(
            "INSERT INTO memories "
            "(created_at, updated_at, category, title, body, tags, source, session_id, pinned, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, now, category, title, body, tags, source, session_id, int(pinned), embedding_json),
        )
        self.conn.commit()
        return cur.lastrowid

    def find_duplicate(
        self,
        category: str,
        title: str,
        body: str = "",
        fuzzy_threshold: float = FUZZY_DUPLICATE_THRESHOLD,
    ) -> sqlite3.Row | None:
        """First tries an exact, case/whitespace-normalized title match (fast, indexed
        SQL, catches the common case of the same fact re-extracted verbatim). If that
        finds nothing and `body` is provided, falls back to word-overlap similarity
        (title+body combined, since titles alone are usually too short to compare
        meaningfully) against same-category candidates, so an obvious rephrasing is
        still caught instead of silently accumulating as a near-duplicate. Pass
        fuzzy_threshold=1.1 (impossible to reach) to disable the fuzzy fallback.
        """
        normalized = title.strip().lower()
        row = self.conn.execute(
            "SELECT * FROM memories WHERE category = ? AND superseded_by IS NULL "
            "AND LOWER(TRIM(title)) = ? LIMIT 1",
            (category, normalized),
        ).fetchone()
        if row is not None:
            return row
        if not body:
            return None

        candidates = self.conn.execute(
            "SELECT * FROM memories WHERE category = ? AND superseded_by IS NULL",
            (category,),
        ).fetchall()
        best_row, best_score = None, 0.0
        for candidate in candidates:
            score = _memory_similarity(title, body, candidate["title"], candidate["body"])
            if score > best_score:
                best_row, best_score = candidate, score
        return best_row if best_score >= fuzzy_threshold else None

    def search(
        self, query: str = "", category: str | None = None, limit: int = 20
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM memories WHERE superseded_by IS NULL"
        params: list = []
        if category:
            sql += " AND category = ?"
            params.append(category)
        if query:
            sql += " AND (title LIKE ? OR body LIKE ? OR tags LIKE ?)"
            like = f"%{query}%"
            params += [like, like, like]
        sql += " ORDER BY pinned DESC, created_at DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def recent_by_category(self, category: str, limit: int = 3) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM memories WHERE category = ? AND superseded_by IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (category, limit),
        ).fetchall()

    def pinned(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM memories WHERE pinned = 1 AND superseded_by IS NULL "
            "ORDER BY created_at DESC"
        ).fetchall()

    def all_active(self, limit: int = 10000) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM memories WHERE superseded_by IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def forget(self, memory_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def supersede(self, old_id: int, new_id: int) -> bool:
        """Marks `old_id` as replaced by `new_id`, retiring it from all_active()/
        search()/context injection without deleting it (audit trail preserved).
        """
        cur = self.conn.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE id = ?",
            (new_id, _now(), old_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def start_session(self, session_id: str) -> None:
        self.conn.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?, ?)", (session_id, _now())
        )
        self.conn.commit()

    def end_session(self, session_id: str, task_summary: str = "") -> None:
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, task_summary = ? WHERE id = ?",
            (_now(), task_summary, session_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
