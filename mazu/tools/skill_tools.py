from mazu.skills.manager import SkillManager
from mazu.tools.base import Tool, ToolResult


def make_skill_tools(manager: SkillManager) -> list[Tool]:
    def save_skill(input: dict) -> ToolResult:
        try:
            manager.save(
                name=input["name"],
                description=input["description"],
                code=input["code"],
                tags=input.get("tags", ""),
            )
            return ToolResult(
                f"Saved skill '{input['name']}'. It can now be run directly with run_skill "
                "instead of re-deriving the logic."
            )
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    def list_skills(input: dict) -> ToolResult:
        metas = manager.list()
        if not metas:
            return ToolResult("No skills saved yet.")
        lines = [
            f"- {m['name']}: {m['description']} "
            f"(used {m.get('usage_count', 0)}x, tags: {m.get('tags') or '-'})"
            for m in metas
        ]
        return ToolResult("\n".join(lines))

    def run_skill(input: dict) -> ToolResult:
        output, is_error = manager.run(input["name"], input.get("args", {}))
        return ToolResult(output, is_error=is_error)

    return [
        Tool(
            name="save_skill",
            description=(
                "Save a reusable Python function as a named local skill, so a similar task in "
                "the future can be solved by running it directly instead of re-deriving the "
                "logic from scratch. Use this after solving a non-trivial, generally-applicable "
                "problem (e.g. a specific parsing routine, a repeatable check, a data "
                "transformation) — not for one-off or trivial logic. `code` must define "
                "exactly one function: `def run(args: dict) -> str:`."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique identifier, e.g. 'parse_apache_log'",
                    },
                    "description": {
                        "type": "string",
                        "description": "What it does and when to use it",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python source defining `def run(args: dict) -> str:`",
                    },
                    "tags": {"type": "string"},
                },
                "required": ["name", "description", "code"],
            },
            handler=save_skill,
        ),
        Tool(
            name="list_skills",
            description="List all locally saved skills available for this project.",
            input_schema={"type": "object", "properties": {}},
            handler=list_skills,
        ),
        Tool(
            name="run_skill",
            description="Execute a previously saved skill by name with the given arguments.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args": {
                        "type": "object",
                        "description": "Arguments passed to the skill's run(args) function",
                    },
                },
                "required": ["name"],
            },
            handler=run_skill,
            destructive=True,
        ),
    ]
