from mazu.curator.analysis.conflicts import find_candidate_pairs
from mazu.curator.store import CuratorStore
from mazu.memory.store import MemoryStore
from mazu.tools.base import Tool, ToolResult

AREA = "conflicts"

_VALID_KINDS = {"contradiction", "refinement", "unrelated"}


def make_curator_conflict_tools(
    memory_store: MemoryStore, curator_store: CuratorStore, run_id: str, dry_run: bool
) -> list[Tool]:
    def find_related_memory_pairs(input: dict) -> ToolResult:
        pairs = find_candidate_pairs(memory_store, max_pairs=input.get("max_pairs", 20))
        if not pairs:
            return ToolResult("No related-but-not-duplicate memory pairs found.")
        return ToolResult("\n\n".join(
            f"A: [{a['id']}] ({a['category']}) {a['title']}\n   {a['body'][:200]}\n"
            f"B: [{b['id']}] ({b['category']}) {b['title']}\n   {b['body'][:200]}"
            for a, b in pairs
        ))

    def record_conflict(input: dict) -> ToolResult:
        kind = input["kind"]
        if kind not in _VALID_KINDS:
            return ToolResult(f"kind must be one of {sorted(_VALID_KINDS)}.", is_error=True)
        # find_related_memory_pairs (this tool's own read-only companion above)
        # only ever surfaces pairs of memories that genuinely exist, so a real
        # model following the intended workflow can never trigger this -- but
        # nothing structurally stopped a stray/hallucinated id from being
        # recorded, leaving memory_conflicts with a dangling reference that
        # would confuse a later resolve_conflict or `mazu curator conflicts`
        # listing. Cheap to validate, so validated.
        if memory_store.get(int(input["memory_id_a"])) is None or memory_store.get(int(input["memory_id_b"])) is None:
            return ToolResult("Both memory_id_a and memory_id_b must refer to existing memories.", is_error=True)
        rationale = input["rationale"]
        if dry_run:
            return ToolResult(f"(dry-run) would record a '{kind}' between {input['memory_id_a']} and {input['memory_id_b']}")
        conflict_id = curator_store.record_conflict(
            memory_id_a=int(input["memory_id_a"]), memory_id_b=int(input["memory_id_b"]),
            kind=kind, rationale=rationale, confidence=input.get("confidence"),
        )
        curator_store.log_entry(
            run_id=run_id, area=AREA, action="record_conflict", target_type="memory_conflict",
            target_id=str(conflict_id), rationale=rationale, applied=True, outcome="ok",
        )
        return ToolResult(f"Recorded conflict {conflict_id} ({kind}).")

    def resolve_conflict(input: dict) -> ToolResult:
        conflict_id = int(input["conflict_id"])
        resolution = input["resolution"]
        if dry_run:
            return ToolResult(f"(dry-run) would resolve conflict {conflict_id}")
        ok = curator_store.resolve_conflict(conflict_id, resolution)
        if not ok:
            return ToolResult(f"No conflict with id {conflict_id}.", is_error=True)
        curator_store.log_entry(
            run_id=run_id, area=AREA, action="resolve_conflict", target_type="memory_conflict",
            target_id=str(conflict_id), rationale=resolution, applied=True, outcome="ok",
        )
        return ToolResult(f"Resolved conflict {conflict_id}: {resolution}")

    def list_conflicts(input: dict) -> ToolResult:
        rows = curator_store.list_conflicts(unresolved_only=input.get("unresolved_only", True))
        if not rows:
            return ToolResult("(none)")
        return ToolResult("\n".join(
            f"[{r['id']}] {r['kind']} between {r['memory_id_a']} and {r['memory_id_b']}: {r['rationale']}"
            for r in rows
        ))

    return [
        Tool(
            name="find_related_memory_pairs",
            description="Find pairs of active memories that are related but not near-duplicates (read-only).",
            input_schema={"type": "object", "properties": {"max_pairs": {"type": "integer"}}},
            handler=find_related_memory_pairs,
        ),
        Tool(
            name="record_conflict",
            description=(
                "Record a judgment about a memory pair: 'contradiction' (they disagree), "
                "'refinement' (B is a more precise/updated version of A), or 'unrelated' "
                "(looked related but aren't). Use edit_memory/supersede_memory/archive_memory "
                "(from the memory toolset) to actually resolve a contradiction once found, then "
                "call resolve_conflict to close it out."
            ),
            input_schema={"type": "object", "properties": {
                "memory_id_a": {"type": "integer"}, "memory_id_b": {"type": "integer"},
                "kind": {"type": "string"}, "rationale": {"type": "string"},
                "confidence": {"type": "number"},
            }, "required": ["memory_id_a", "memory_id_b", "kind", "rationale"]},
            handler=record_conflict,
        ),
        Tool(
            name="resolve_conflict",
            description="Mark a recorded conflict as resolved, after acting on it (or deciding not to).",
            input_schema={"type": "object", "properties": {
                "conflict_id": {"type": "integer"}, "resolution": {"type": "string"},
            }, "required": ["conflict_id", "resolution"]},
            handler=resolve_conflict,
        ),
        Tool(
            name="list_conflicts", description="List recorded conflicts (unresolved by default).",
            input_schema={"type": "object", "properties": {"unresolved_only": {"type": "boolean"}}},
            handler=list_conflicts,
        ),
    ]
