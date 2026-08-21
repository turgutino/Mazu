from mazu.curator.store import CuratorStore
from mazu.memory.consolidate import find_duplicate_clusters
from mazu.memory.staleness import find_stale_candidates
from mazu.memory.store import MemoryStore
from mazu.tools.base import Tool, ToolResult


def _row_summary(row) -> str:
    return f"[{row['id']}] ({row['category']}) {row['title']}"


def make_curator_memory_tools(
    store: MemoryStore, curator_store: CuratorStore, run_id: str, dry_run: bool, area: str = "memory"
) -> list[Tool]:
    """Every write tool here requires a `rationale` field in its input schema, so
    Curator's diary (curator_log) is populated by construction -- the model can't
    make a change without also explaining it. dry_run=True swaps every mutating
    call for a no-op that reports what *would* happen and logs with applied=0,
    while still running the exact same reasoning/tool-call sequence, so `mazu
    curator run --dry-run` is a faithful preview, not a different code path.

    `area` -- which orchestrator area's diary these calls get attributed to.
    Regression fix: this used to be a hardcoded module-level "memory" constant,
    which was correct when this factory was only ever used by the memory area
    itself, but the SAME factory is also registered for the conflicts area (so
    a real contradiction can actually be resolved via edit/supersede/archive,
    not just recorded). With the hardcoded constant, every such resolution made
    during a conflicts pass got silently mis-tagged area='memory' in curator_log
    -- invisible to `mazu curator log --area conflicts`, and undercounted in that
    run's own "N decision(s)" summary. Found live: a real conflicts pass that
    genuinely called supersede_memory to resolve a contradiction showed up under
    `--area memory` instead. Now the caller (mazu/curator/tools/registry.py)
    always passes the real area explicitly.
    """

    def _log(action, target_id, rationale, reversal_hint=None, outcome="ok"):
        curator_store.log_entry(
            run_id=run_id, area=area, action=action, target_type="memory",
            target_id=str(target_id), rationale=rationale, reversal_hint=reversal_hint,
            applied=not dry_run, outcome=outcome,
        )

    def recall_memories(input: dict) -> ToolResult:
        rows = store.search(query=input.get("query", ""), category=input.get("category"), limit=input.get("limit", 20))
        if not rows:
            return ToolResult("No matching memories.")
        return ToolResult("\n".join(f"{_row_summary(r)}\n  {r['body'][:300]}" for r in rows))

    def list_memories(input: dict) -> ToolResult:
        rows = store.list_archived(limit=200) if input.get("archived") else store.all_active(limit=500)
        if not rows:
            return ToolResult("(none)")
        return ToolResult("\n".join(_row_summary(r) for r in rows))

    def memory_stats(input: dict) -> ToolResult:
        s = store.stats()
        return ToolResult(
            f"total={s['total']} active={s['active']} superseded={s['superseded']} "
            f"archived={s['archived']} pinned={s['pinned']}"
        )

    def find_duplicate_memories(input: dict) -> ToolResult:
        clusters = find_duplicate_clusters(store, threshold=input.get("threshold", 0.5))
        if not clusters:
            return ToolResult("No near-duplicate clusters found.")
        lines = []
        for cluster in clusters:
            lines.append(", ".join(_row_summary(r) for r in cluster))
        return ToolResult("\n".join(lines))

    def find_stale_memories(input: dict) -> ToolResult:
        candidates = find_stale_candidates(store, min_days=input.get("min_days", 30))
        if not candidates:
            return ToolResult("No stale candidates.")
        return ToolResult("\n".join(_row_summary(r) for r in candidates))

    def add_memory(input: dict) -> ToolResult:
        rationale = input["rationale"]
        if dry_run:
            _log("add_memory", "(new)", rationale, reversal_hint="mazu memory forget <new id>")
            return ToolResult(f"(dry-run) would add memory: {input['title']}")
        memory_id = store.add(
            category=input["category"], title=input["title"], body=input["body"],
            tags=input.get("tags", ""), source="auto_extracted",
        )
        _log("add_memory", memory_id, rationale, reversal_hint=f"mazu memory forget {memory_id}")
        return ToolResult(f"Added memory {memory_id}: {input['title']}")

    def edit_memory(input: dict) -> ToolResult:
        rationale = input["rationale"]
        memory_id = int(input["memory_id"])
        existing = store.get(memory_id)
        if existing is None:
            return ToolResult(f"No memory with id {memory_id}.", is_error=True)
        old_title, old_body = existing["title"], existing["body"]
        if dry_run:
            _log("edit_memory", memory_id, rationale)
            return ToolResult(f"(dry-run) would edit memory {memory_id}")
        store.edit(memory_id, title=input.get("title"), body=input.get("body"))
        reversal = f'mazu memory edit {memory_id} --title "{old_title}" --body "{old_body}"'
        _log("edit_memory", memory_id, rationale, reversal_hint=reversal)
        return ToolResult(f"Edited memory {memory_id}.")

    def archive_memory(input: dict) -> ToolResult:
        rationale = input["rationale"]
        memory_id = int(input["memory_id"])
        if dry_run:
            _log("archive_memory", memory_id, rationale, reversal_hint=f"mazu memory unarchive {memory_id}")
            return ToolResult(f"(dry-run) would archive memory {memory_id}")
        ok = store.archive(memory_id)
        if not ok:
            return ToolResult(f"No memory with id {memory_id}.", is_error=True)
        _log("archive_memory", memory_id, rationale, reversal_hint=f"mazu memory unarchive {memory_id}")
        return ToolResult(f"Archived memory {memory_id}.")

    def unarchive_memory(input: dict) -> ToolResult:
        rationale = input["rationale"]
        memory_id = int(input["memory_id"])
        if dry_run:
            _log("unarchive_memory", memory_id, rationale)
            return ToolResult(f"(dry-run) would unarchive memory {memory_id}")
        ok = store.unarchive(memory_id)
        if not ok:
            return ToolResult(f"No memory with id {memory_id}.", is_error=True)
        _log("unarchive_memory", memory_id, rationale)
        return ToolResult(f"Unarchived memory {memory_id}.")

    def pin_memory(input: dict) -> ToolResult:
        rationale = input["rationale"]
        memory_id = int(input["memory_id"])
        if dry_run:
            _log("pin_memory", memory_id, rationale, reversal_hint=f"mazu memory unpin {memory_id}")
            return ToolResult(f"(dry-run) would pin memory {memory_id}")
        ok = store.pin(memory_id)
        if not ok:
            return ToolResult(f"No memory with id {memory_id}.", is_error=True)
        _log("pin_memory", memory_id, rationale, reversal_hint=f"mazu memory unpin {memory_id}")
        return ToolResult(f"Pinned memory {memory_id}.")

    def unpin_memory(input: dict) -> ToolResult:
        rationale = input["rationale"]
        memory_id = int(input["memory_id"])
        if dry_run:
            _log("unpin_memory", memory_id, rationale, reversal_hint=f"mazu memory pin {memory_id}")
            return ToolResult(f"(dry-run) would unpin memory {memory_id}")
        ok = store.unpin(memory_id)
        if not ok:
            return ToolResult(f"No memory with id {memory_id}.", is_error=True)
        _log("unpin_memory", memory_id, rationale, reversal_hint=f"mazu memory pin {memory_id}")
        return ToolResult(f"Unpinned memory {memory_id}.")

    def supersede_memory(input: dict) -> ToolResult:
        rationale = input["rationale"]
        old_id, new_id = int(input["old_id"]), int(input["new_id"])
        if store.get(old_id) is None or store.get(new_id) is None:
            return ToolResult("Both old_id and new_id must exist.", is_error=True)
        if dry_run:
            _log("supersede_memory", old_id, rationale)
            return ToolResult(f"(dry-run) would supersede {old_id} with {new_id}")
        store.supersede(old_id, new_id)
        _log(
            "supersede_memory", old_id, rationale,
            reversal_hint=f"no direct undo command yet -- use 'mazu memory edit {old_id}' to restore its content if needed",
        )
        return ToolResult(f"Memory {old_id} marked as superseded by {new_id}.")

    _rationale_prop = {"rationale": {"type": "string", "description": "Why this change is being made -- recorded in Curator's diary."}}

    return [
        Tool(
            name="recall_memories", description="Search active memories by text/category.",
            input_schema={"type": "object", "properties": {
                "query": {"type": "string"}, "category": {"type": "string"}, "limit": {"type": "integer"},
            }},
            handler=recall_memories,
        ),
        Tool(
            name="list_memories", description="List all active memories, or all archived ones if archived=true.",
            input_schema={"type": "object", "properties": {"archived": {"type": "boolean"}}},
            handler=list_memories,
        ),
        Tool(
            name="memory_stats", description="Summary counts: total/active/superseded/archived/pinned.",
            input_schema={"type": "object", "properties": {}},
            handler=memory_stats,
        ),
        Tool(
            name="find_duplicate_memories", description="Find near-duplicate memory clusters (read-only).",
            input_schema={"type": "object", "properties": {"threshold": {"type": "number"}}},
            handler=find_duplicate_memories,
        ),
        Tool(
            name="find_stale_memories", description="Find memories unused for min_days+ (read-only).",
            input_schema={"type": "object", "properties": {"min_days": {"type": "integer"}}},
            handler=find_stale_memories,
        ),
        Tool(
            name="add_memory", description="Add a new memory (e.g. a corrected/refined fact).",
            input_schema={"type": "object", "properties": {
                "category": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"},
                "tags": {"type": "string"}, **_rationale_prop,
            }, "required": ["category", "title", "body", "rationale"]},
            handler=add_memory,
        ),
        Tool(
            name="edit_memory", description="Edit a memory's title/body in place.",
            input_schema={"type": "object", "properties": {
                "memory_id": {"type": "integer"}, "title": {"type": "string"}, "body": {"type": "string"}, **_rationale_prop,
            }, "required": ["memory_id", "rationale"]},
            handler=edit_memory,
        ),
        Tool(
            name="archive_memory", description="Archive a stale/low-value memory. Reversible.",
            input_schema={"type": "object", "properties": {"memory_id": {"type": "integer"}, **_rationale_prop},
                          "required": ["memory_id", "rationale"]},
            handler=archive_memory,
        ),
        Tool(
            name="unarchive_memory", description="Restore an archived memory to active use.",
            input_schema={"type": "object", "properties": {"memory_id": {"type": "integer"}, **_rationale_prop},
                          "required": ["memory_id", "rationale"]},
            handler=unarchive_memory,
        ),
        Tool(
            name="pin_memory", description="Pin a memory so it's always included in context.",
            input_schema={"type": "object", "properties": {"memory_id": {"type": "integer"}, **_rationale_prop},
                          "required": ["memory_id", "rationale"]},
            handler=pin_memory,
        ),
        Tool(
            name="unpin_memory", description="Unpin a memory.",
            input_schema={"type": "object", "properties": {"memory_id": {"type": "integer"}, **_rationale_prop},
                          "required": ["memory_id", "rationale"]},
            handler=unpin_memory,
        ),
        Tool(
            name="supersede_memory", description="Mark old_id as replaced by new_id.",
            input_schema={"type": "object", "properties": {
                "old_id": {"type": "integer"}, "new_id": {"type": "integer"}, **_rationale_prop,
            }, "required": ["old_id", "new_id", "rationale"]},
            handler=supersede_memory,
        ),
    ]
