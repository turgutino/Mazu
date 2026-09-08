from mazu.memory.embeddings import embed_text
from mazu.memory.store import MemoryStore, _memory_similarity
from mazu.tools.base import Tool, ToolResult

CATEGORIES = ["decision", "convention", "mistake", "task_outcome", "fact", "user_preference"]


def _observe_explicit_remember(
    belief_log,
    project: str | None,
    session_id: str,
    category: str,
    memory_id: int,
    input: dict,
    target_store: MemoryStore,
    supersedes_id: int | None,
    old_row,
    tool_message: str,
    tool_is_error: bool,
) -> None:
    """SHADOW-ONLY observer for a real `remember` tool call, already completed by
    the time this runs. Zero LLM calls: an explicit supersede already tells us the
    relationship (the agent asserted it by acting on it -- no need to guess), and
    a plain insert only gets a local, zero-cost gray-zone lookup (reusing
    find_gray_zone_match, the same Jaccard check the auto-extraction shadow path
    already uses) to note whether a similar memory existed that the agent chose
    NOT to touch. Never raises past its own try/except in the caller.
    """
    if supersedes_id is not None:
        similarity = None
        if old_row is not None:
            similarity = _memory_similarity(
                old_row["title"], old_row["body"], input["title"], input["body"]
            )
        belief_log.log_explicit_action(
            session_id=session_id,
            project=project,
            action_type="supersede",
            category=category,
            new_memory_id=memory_id,
            new_title=input["title"],
            new_body=input["body"],
            old_memory_id=supersedes_id,
            old_title=old_row["title"] if old_row is not None else None,
            old_body=old_row["body"] if old_row is not None else None,
            similarity_score=similarity,
            # The agent itself chose to supersede -- that IS the asserted
            # relationship, not a guess. Never computed by a classifier.
            asserted_relationship="replacement",
            tool_result=tool_message,
            tool_success=not tool_is_error,
        )
        return

    # Plain insert: local-only gray-zone lookup against what's now active in the
    # same category (excluding the row we just inserted). No classifier call.
    from mazu.memory.belief import find_gray_zone_match

    nearest_row, nearest_score = find_gray_zone_match(
        target_store, category, input["title"], input["body"]
    )
    belief_log.log_explicit_action(
        session_id=session_id,
        project=project,
        action_type="insert",
        category=category,
        new_memory_id=memory_id,
        new_title=input["title"],
        new_body=input["body"],
        nearest_active_memory_id=nearest_row["id"] if nearest_row is not None else None,
        nearest_active_title=nearest_row["title"] if nearest_row is not None else None,
        nearest_active_similarity=nearest_score if nearest_row is not None else None,
        tool_result=tool_message,
        tool_success=not tool_is_error,
    )


def make_memory_tools(
    store: MemoryStore,
    global_store: MemoryStore,
    session_id: str,
    belief_log=None,
    project: str | None = None,
    dry_run: bool = False,
) -> list[Tool]:
    def remember(input: dict) -> ToolResult:
        try:
            category = input["category"]
            # user_preference is about the person, not this codebase -- it goes in the
            # global store so it's visible in every project, not just this one.
            target_store = global_store if category == "user_preference" else store
            scope = "global, all projects" if category == "user_preference" else "this project"
            supersedes_id = input.get("supersedes_id")
            if dry_run:
                # Real bug found live: `mazu run --dry-run` never actually reached
                # here (this tool had no dry_run concept at all), so `remember`
                # made a real, permanent write to memory.db even during a run
                # explicitly meant to change nothing -- directly contradicting its
                # own "nothing was actually changed" report. Mirrors write_file/
                # edit_file's own dry-run pattern (mazu/tools/fs.py): report what
                # would happen, touch nothing.
                note = f" (would mark memory {supersedes_id} as superseded)" if supersedes_id is not None else ""
                return ToolResult(f"[dry-run] Would remember ({scope}): {input['title']}{note}")
            # None unless MAZU_SEMANTIC_MEMORY is opted into (see memory/embeddings.py)
            # -- add() stores None as a plain NULL, so this never changes behavior for
            # anyone who hasn't turned semantic search on.
            embedding = embed_text(f"{input['title']} {input['body']}")
            # Fetched BEFORE add()/supersede() so the shadow observer sees the old
            # memory's content exactly as it was when the agent decided to replace
            # it -- purely a read, changes nothing.
            old_row = (
                target_store.get(int(supersedes_id)) if supersedes_id is not None else None
            )
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
            if supersedes_id is not None:
                if target_store.supersede(int(supersedes_id), memory_id):
                    note = f" (marked memory {supersedes_id} as superseded)"
                else:
                    note = f" (warning: no memory with id {supersedes_id} to supersede)"
            message = f"Remembered (id={memory_id}, {scope}): {input['title']}{note}"

            if belief_log is not None:
                try:
                    _observe_explicit_remember(
                        belief_log, project, session_id, category, memory_id, input,
                        target_store, supersedes_id, old_row, message, False,
                    )
                except Exception:
                    # Shadow observation must never affect the real, already-completed
                    # remember/supersede action above -- swallow silently, same
                    # "best-effort, never blocks the real thing" contract as the
                    # auto-extraction shadow path.
                    pass

            return ToolResult(message)
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
