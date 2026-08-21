from mazu.curator.store import CuratorStore
from mazu.tools.base import Tool, ToolResult


def make_diary_tools(curator_store: CuratorStore, run_id: str, area: str) -> list[Tool]:
    """`curator_note` is always registered, in every area's pass -- it's how Curator
    records "I looked at this and deliberately decided not to act", which matters
    just as much for the self-report as every actual mutation does (a silent pass
    that changed nothing looks identical to a pass that never really looked)."""

    def curator_note(input: dict) -> ToolResult:
        curator_store.log_entry(
            run_id=run_id, area=area, action="note",
            rationale=input["note"], outcome="ok",
        )
        return ToolResult("Noted.")

    return [
        Tool(
            name="curator_note",
            description=(
                "Record an observation or decision in Curator's diary, including "
                "'considered X and decided not to act' -- use this whenever you "
                "reviewed something but chose not to change it, so the self-report "
                "reflects what you actually looked at, not just what you changed."
            ),
            input_schema={
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
            handler=curator_note,
        ),
    ]
