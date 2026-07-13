CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    category        TEXT NOT NULL CHECK (category IN ('decision', 'convention', 'mistake', 'task_outcome', 'fact', 'user_preference')),
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    tags            TEXT,
    source          TEXT NOT NULL CHECK (source IN ('explicit', 'auto_extracted')),
    session_id      TEXT,
    relevance_score REAL NOT NULL DEFAULT 1.0,
    superseded_by   INTEGER REFERENCES memories(id),
    pinned          INTEGER NOT NULL DEFAULT 0,
    -- JSON-encoded embedding vector, NULL unless semantic search is opted into via
    -- MAZU_SEMANTIC_MEMORY (see mazu/memory/embeddings.py) at write time.
    embedding       TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned);
CREATE INDEX IF NOT EXISTS idx_memories_superseded_created ON memories(superseded_by, created_at);

CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    task_summary TEXT
);
