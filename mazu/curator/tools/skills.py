from mazu.action_log.store import ActionLogStore
from mazu.curator.store import CuratorStore
from mazu.skills.manager import SkillManager
from mazu.tools.base import Tool, ToolResult

AREA = "skills"


def make_curator_skill_tools(
    manager: SkillManager,
    curator_store: CuratorStore,
    run_id: str,
    dry_run: bool,
    action_log_store: ActionLogStore | None = None,
) -> list[Tool]:
    """No run_skill here, deliberately -- executing arbitrary saved local Python
    isn't curation authority (see CURATOR_FORBIDDEN_TOOLS). Every write tool
    requires a `rationale`, same as the memory tools."""

    def _log(action, target_id, rationale, reversal_hint=None):
        curator_store.log_entry(
            run_id=run_id, area=AREA, action=action, target_type="skill",
            target_id=target_id, rationale=rationale, reversal_hint=reversal_hint,
            applied=not dry_run, outcome="ok",
        )

    def list_skills(input: dict) -> ToolResult:
        metas = manager.list(include_archived=input.get("archived", False))
        if not metas:
            return ToolResult("(none)")
        return ToolResult("\n".join(
            f"{m['name']}: {m['description']} (used {m.get('usage_count', 0)}x, "
            f"success={m.get('success_count', 0)} failure={m.get('failure_count', 0)})"
            for m in metas
        ))

    def read_skill(input: dict) -> ToolResult:
        name = input["name"]
        meta = manager.get_meta(name)
        code = manager.read_code(name)
        if meta is None or code is None:
            return ToolResult(f"No skill named '{name}'.", is_error=True)
        return ToolResult(f"meta: {meta}\n\ncode:\n{code}")

    def skill_outcomes(input: dict) -> ToolResult:
        if action_log_store is None:
            return ToolResult("No action log available.")
        outcomes = action_log_store.skill_run_outcomes(since_days=input.get("since_days"))
        if not outcomes:
            return ToolResult("No run_skill history yet.")
        return ToolResult("\n".join(f"{name}: {counts}" for name, counts in outcomes.items()))

    def write_skill(input: dict) -> ToolResult:
        rationale = input["rationale"]
        name = input["name"]
        if dry_run:
            _log("write_skill", name, rationale)
            return ToolResult(f"(dry-run) would write skill '{name}'")
        try:
            manager.save(name, input["description"], input["code"], tags=input.get("tags", ""))
        except ValueError as e:
            return ToolResult(str(e), is_error=True)
        manager.update_meta(
            name, curator_revision=(manager.get_meta(name) or {}).get("curator_revision", 0) + 1,
        )
        _log("write_skill", name, rationale, reversal_hint="no automatic undo -- inspect with 'mazu skills list' / re-save manually if needed")
        return ToolResult(f"Wrote skill '{name}'.")

    def archive_skill(input: dict) -> ToolResult:
        rationale = input["rationale"]
        name = input["name"]
        if dry_run:
            _log("archive_skill", name, rationale, reversal_hint=f"mazu skills unarchive {name}")
            return ToolResult(f"(dry-run) would archive skill '{name}'")
        ok = manager.archive(name, reason=rationale)
        if not ok:
            return ToolResult(f"No skill named '{name}'.", is_error=True)
        _log("archive_skill", name, rationale, reversal_hint=f"mazu skills unarchive {name}")
        return ToolResult(f"Archived skill '{name}'.")

    def unarchive_skill(input: dict) -> ToolResult:
        rationale = input["rationale"]
        name = input["name"]
        if dry_run:
            _log("unarchive_skill", name, rationale)
            return ToolResult(f"(dry-run) would unarchive skill '{name}'")
        ok = manager.unarchive(name)
        if not ok:
            return ToolResult(f"No skill named '{name}'.", is_error=True)
        _log("unarchive_skill", name, rationale)
        return ToolResult(f"Unarchived skill '{name}'.")

    def supersede_skill(input: dict) -> ToolResult:
        rationale = input["rationale"]
        old_name, new_name = input["old_name"], input["new_name"]
        if dry_run:
            _log("supersede_skill", old_name, rationale)
            return ToolResult(f"(dry-run) would supersede '{old_name}' with '{new_name}'")
        ok = manager.supersede(old_name, new_name)
        if not ok:
            return ToolResult("Both old_name and new_name must exist.", is_error=True)
        _log("supersede_skill", old_name, rationale, reversal_hint=f"mazu skills unarchive {old_name}")
        return ToolResult(f"Skill '{old_name}' marked as superseded by '{new_name}'.")

    _rationale_prop = {"rationale": {"type": "string", "description": "Why this change is being made -- recorded in Curator's diary."}}

    return [
        Tool(
            name="list_skills", description="List active skills, or all (including archived) if archived=true.",
            input_schema={"type": "object", "properties": {"archived": {"type": "boolean"}}},
            handler=list_skills,
        ),
        Tool(
            name="read_skill", description="Read a skill's full metadata and code.",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            handler=read_skill,
        ),
        Tool(
            name="skill_outcomes",
            description="Per-skill success/error counts from real run_skill history (read-only ground truth).",
            input_schema={"type": "object", "properties": {"since_days": {"type": "integer"}}},
            handler=skill_outcomes,
        ),
        Tool(
            name="write_skill", description="Create or rewrite a skill's code (e.g. to fix a bug in a failing skill).",
            input_schema={"type": "object", "properties": {
                "name": {"type": "string"}, "description": {"type": "string"}, "code": {"type": "string"},
                "tags": {"type": "string"}, **_rationale_prop,
            }, "required": ["name", "description", "code", "rationale"]},
            handler=write_skill,
        ),
        Tool(
            name="archive_skill", description="Archive a failing/unused skill. Reversible.",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}, **_rationale_prop},
                          "required": ["name", "rationale"]},
            handler=archive_skill,
        ),
        Tool(
            name="unarchive_skill", description="Restore an archived skill.",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}, **_rationale_prop},
                          "required": ["name", "rationale"]},
            handler=unarchive_skill,
        ),
        Tool(
            name="supersede_skill", description="Mark old_name as replaced by new_name (merges near-duplicates).",
            input_schema={"type": "object", "properties": {
                "old_name": {"type": "string"}, "new_name": {"type": "string"}, **_rationale_prop,
            }, "required": ["old_name", "new_name", "rationale"]},
            handler=supersede_skill,
        ),
    ]
