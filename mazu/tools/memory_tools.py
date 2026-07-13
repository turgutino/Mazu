from mazu.memory.embeddings import embed_text
from mazu.memory.store import MemoryStore
from mazu.tools.base import Tool, ToolResult

CATEGORIES = ["decision", "convention", "mistake", "task_outcome", "fact", "user_preference"]


def make_memory_tools(store: MemoryStore, global_store: MemoryStore, session_id: str) -> list[Tool]:
    def remember(input: dict) -> ToolResult:
        try:
            category = input["category"]
            # user_preference is about the person, not this codebase -- it goes in the
            # global store so it's visible in every project, not just this one.
            target_store = global_store if category == "user_preference" else store
            # None unless MAZU_SEMANTIC_MEMORY is opted into (see memory/embeddings.py)
            # -- add() stores None as a plain NULL, so this never changes behavior for
            # anyone who hasn't turned semantic search on.
            embedding = embed_text(f"{input['title']} {input['body']}")
            memory_id = target_store.add(
                category=category,
                title=input["title"],
                body=input["body"],
                tags=input.get("tags", ""),
                source="explicit",
                session_id=session_id,
                embedding=embedding,
            )
            note = ""
            supersedes_id = input.get("supersedes_id")
            if supersedes_id is not None:
                if target_store.supersede(int(supersedes_id), memory_id):
                    note = f" (marked memory {supersedes_id} as superseded)"
                else:
                    note = f" (warning: no memory with id {supersedes_id} to supersede)"
            scope = "global, all projects" if category == "user_preference" else "this project"
            return ToolResult(f"Remembered (id={memory_id}, {scope}): {input['title']}{note}")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    def recall(input: dict) -> ToolResult:
        try:
            rows = store.search(query=input.get("query", ""), category=input.get("category"))
            if not rows:
                return ToolResult("No matching memories found.")
            lines = [
                f"[{r['id']}] ({r['category']}) {r['title']}: {r['body']} (tags: {r['tags'] or '-'})"
                for r in rows
            ]
            return ToolResult("\n".join(lines))
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    return [
        Tool(
            name="remember",
            description=(
                "Store a durable fact, decision, coding convention, or lesson-learned that "
                "should persist across sessions and be recalled automatically in future work "
                "on this project. Use this proactively when you make an architectural "
                "decision, discover a project-specific convention, identify a mistake or "
                "gotcha to avoid, or complete/fail a notable task. Use category "
                "'user_preference' specifically for durable facts about the person you're "
                "working with (their name, preferred language, experience level, general "
                "working style) — those are stored globally and recalled in every project, "
                "not just this one."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "title": {"type": "string", "description": "One-line summary"},
                    "body": {
                        "type": "string",
                        "description": "Full detail — the why, not just the what",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags, e.g. 'auth,api'",
                    },
                    "supersedes_id": {
                        "type": "integer",
                        "description": (
                            "If this replaces an earlier memory you already know about "
                            "(e.g. shown in Project Memory or from recall) — a decision that "
                            "changed, a convention that was updated — provide that memory's "
                            "id here so it gets retired instead of left active alongside the "
                            "new one."
                        ),
                    },
                },
                "required": ["category", "title", "body"],
            },
            handler=remember,
        ),
        Tool(
            name="recall",
            description=(
                "Search project memory for facts, decisions, conventions, or past mistakes "
                "related to a query. Use this when you need something specific that may not "
                "have been auto-loaded at session start."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                },
            },
            handler=recall,
        ),
    ]
