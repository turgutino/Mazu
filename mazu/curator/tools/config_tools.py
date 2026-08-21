from mazu.config import list_config, set_config_value
from mazu.curator.store import CuratorStore
from mazu.tools.base import Tool, ToolResult

# Hard allowlist, enforced here regardless of what mazu.config.KNOWN_CONFIG_KEYS
# permits -- Curator must never be able to change its OWN model/key/enablement
# (that would let a runaway pass escalate itself), any *_api_key, or
# local_base_url. Checked in the handler, not just documented, so a future
# KNOWN_CONFIG_KEYS addition can't silently widen what Curator can touch.
CURATOR_SETTABLE_CONFIG_KEYS = frozenset({
    "default_model", "router_suggestions", "council_models", "council_lead",
})


def make_curator_config_tools(
    curator_store: CuratorStore, run_id: str, dry_run: bool, area: str = "housekeeping"
) -> list[Tool]:
    """`area` -- see the identical fix/rationale in mazu/curator/tools/memory.py's
    make_curator_memory_tools: this factory is only ever registered under the
    'housekeeping' orchestrator area, but used to hardcode a mismatched 'config'
    tag on every log_entry, making set_config actions invisible to `mazu curator
    log --area housekeeping` and undercounted in that area's own decision count."""

    def _log(action, target_id, rationale, reversal_hint=None):
        curator_store.log_entry(
            run_id=run_id, area=area, action=action, target_type="config",
            target_id=target_id, rationale=rationale, reversal_hint=reversal_hint,
            applied=not dry_run, outcome="ok",
        )

    def get_config(input: dict) -> ToolResult:
        values = list_config()
        if not values:
            return ToolResult("(no config set)")
        return ToolResult("\n".join(f"{k} = {v}" for k, v in sorted(values.items()) if not k.endswith("_api_key")))

    def set_config(input: dict) -> ToolResult:
        key, value, rationale = input["key"], input["value"], input["rationale"]
        if key not in CURATOR_SETTABLE_CONFIG_KEYS:
            return ToolResult(
                f"'{key}' is not one of the config keys Curator is allowed to change "
                f"({', '.join(sorted(CURATOR_SETTABLE_CONFIG_KEYS))}).", is_error=True,
            )
        old_value = list_config().get(key)
        if dry_run:
            _log("set_config", key, rationale)
            return ToolResult(f"(dry-run) would set {key} = {value}")
        set_config_value(key, value)
        reversal = f"mazu config set {key} {old_value}" if old_value is not None else f"mazu config unset {key}"
        _log("set_config", key, rationale, reversal_hint=reversal)
        return ToolResult(f"Set {key} = {value}")

    return [
        Tool(
            name="get_config", description="Show current config values (secrets omitted).",
            input_schema={"type": "object", "properties": {}},
            handler=get_config,
        ),
        Tool(
            name="set_config",
            description=(
                "Change a config value. Only a small allowlist of keys is permitted: "
                f"{', '.join(sorted(CURATOR_SETTABLE_CONFIG_KEYS))}."
            ),
            input_schema={"type": "object", "properties": {
                "key": {"type": "string"}, "value": {"type": "string"},
                "rationale": {"type": "string"},
            }, "required": ["key", "value", "rationale"]},
            handler=set_config,
        ),
    ]
