from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.curator.analysis.branches import find_abandoned_branches
from mazu.curator.analysis.patterns import find_common_tool_sequences
from mazu.runs.store import RunStore
from mazu.tools.base import Tool, ToolResult
from mazu.usage.store import UsageStore


def make_curator_analysis_tools(
    root: Path,
    run_store: RunStore | None,
    usage_store: UsageStore | None,
    action_log_store: ActionLogStore | None,
) -> list[Tool]:
    """All read-only -- this area's whole purpose is surfacing patterns for
    curator_note / the config tools to act on, not mutating runs/checkpoints/
    usage/action-log history directly. RunStore/checkpoints/UsageStore have no
    mutation authority granted to Curator at all (see the plan's access table)."""

    def run_stats(input: dict) -> ToolResult:
        if run_store is None:
            return ToolResult("No run history available.")
        rows = run_store.list_runs(limit=input.get("limit", 50))
        if not rows:
            return ToolResult("No runs recorded yet.")
        return ToolResult("\n".join(
            f"[{r['id']}] {r['task'][:80]!r} model={r['model']} status={r['status']} "
            f"stop_reason={r['stop_reason']}" for r in rows
        ))

    def usage_summary(input: dict) -> ToolResult:
        if usage_store is None:
            return ToolResult("No usage data available.")
        summary = usage_store.summary(since_days=input.get("since_days"))
        if summary["total_calls"] == 0:
            return ToolResult("No usage recorded yet.")
        lines = [f"Total: ${summary['total_cost']:.4f} across {summary['total_calls']} calls"]
        for row in summary["by_model"]:
            lines.append(f"  {row['provider']}:{row['model']} {row['calls']} calls ${row['cost'] or 0:.4f}")
        return ToolResult("\n".join(lines))

    def find_abandoned_branches_tool(input: dict) -> ToolResult:
        candidates = find_abandoned_branches(root, min_age_days=input.get("min_age_days", 14))
        if not candidates:
            return ToolResult("No abandoned branches found.")
        return ToolResult("\n".join(
            f"{c['branch']}: {c['checkpoint_count']} checkpoint(s), last at {c['last_checkpoint_at']}"
            for c in candidates
        ))

    def action_log_patterns(input: dict) -> ToolResult:
        if action_log_store is None:
            return ToolResult("No action log available.")
        patterns = find_common_tool_sequences(action_log_store, min_count=input.get("min_count", 3))
        if not patterns:
            return ToolResult("No recurring tool sequences found.")
        return ToolResult("\n".join(f"{p['sequence'][0]} -> {p['sequence'][1]}: {p['count']}x" for p in patterns))

    return [
        Tool(
            name="run_stats", description="Recent `mazu run` history: task, model, status.",
            input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
            handler=run_stats,
        ),
        Tool(
            name="usage_summary", description="Estimated API spend, grouped by model.",
            input_schema={"type": "object", "properties": {"since_days": {"type": "integer"}}},
            handler=usage_summary,
        ),
        Tool(
            name="find_abandoned_branches",
            description="Git branches with checkpoint history nobody has returned to in a while.",
            input_schema={"type": "object", "properties": {"min_age_days": {"type": "integer"}}},
            handler=find_abandoned_branches_tool,
        ),
        Tool(
            name="action_log_patterns",
            description="Recurring consecutive tool-call sequences across sessions -- a signal for a missing skill.",
            input_schema={"type": "object", "properties": {"min_count": {"type": "integer"}}},
            handler=action_log_patterns,
        ),
    ]
