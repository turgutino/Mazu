import sys
import uuid
from pathlib import Path

import click

from mazu import __version__
from mazu.action_log.store import ActionLogStore
from mazu.agent.autonomous import run_autonomous
from mazu.agent.council import run_council
from mazu.agent.loop import run_chat_loop
from mazu.banner import print_banner
from mazu.checkpoint.manager import CheckpointManager
from mazu.config import (
    _SECRET_CONFIG_KEYS,
    config_path,
    ensure_api_key,
    list_config,
    set_config_value,
    unset_config_value,
)
from mazu.diagnostics import apply_fixes, check_live_api_key, ensure_gitignore, run_diagnostics
from mazu.llm.capabilities import list_capabilities
from mazu.memory.consolidate import apply_consolidation, find_duplicate_clusters
from mazu.memory.retrieval import explain_retrieval
from mazu.memory.store import FUZZY_DUPLICATE_THRESHOLD, MemoryStore
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.tools.fs import make_fs_tools
from mazu.tools.memory_tools import make_memory_tools
from mazu.tools.registry import ToolRegistry
from mazu.tools.shell import make_shell_tool
from mazu.tools.skill_tools import make_skill_tools
from mazu.usage.store import UsageStore

# Model output can contain arbitrary Unicode (arrows, em-dashes, emoji) that the
# default Windows console codepage (cp1252) can't encode, which otherwise crashes
# `print()` mid-response. Force UTF-8 with a safe fallback instead of failing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _mazu_dir(root: Path) -> Path:
    return root / ".mazu"


def _memory_db_path(root: Path) -> Path:
    return _mazu_dir(root) / "memory.db"


def _global_memory_db_path() -> Path:
    # Deliberately not tied to any project root — this is the one store shared across
    # every project, for durable facts about the person, not the codebase.
    return Path.home() / ".mazu" / "global_memory.db"


def _action_log_db_path(root: Path) -> Path:
    # Project-scoped like memory.db, not global like usage.db -- "what did the agent
    # do in this project" is a per-project question, and correlates with checkpoints
    # and project memory writes that are themselves project-scoped.
    return _mazu_dir(root) / "action_log.db"


def _runs_db_path(root: Path) -> Path:
    # Project-scoped like memory.db/action_log.db -- a run's id is a session_id, and
    # session-scoped state is per-project throughout Mazu.
    return _mazu_dir(root) / "runs.db"


def _usage_db_path() -> Path:
    # Global like global_memory.db — spend is tied to the person/API keys, not any
    # one project. A separate file on purpose (see UsageStore's docstring).
    return Path.home() / ".mazu" / "usage.db"


def _build_registry(
    root: Path,
    memory_store: MemoryStore,
    global_memory_store: MemoryStore,
    skill_manager: SkillManager,
    session_id: str,
    dry_run: bool = False,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in make_fs_tools(root, dry_run=dry_run):
        registry.register(tool)
    registry.register(make_shell_tool(root, dry_run=dry_run))
    for tool in make_memory_tools(memory_store, global_memory_store, session_id):
        registry.register(tool)
    for tool in make_skill_tools(skill_manager):
        registry.register(tool)
    return registry


def _parse_shell_allowlist(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return names or None


@click.group()
@click.version_option(version=__version__, prog_name="mazu")
def main() -> None:
    """Mazu — a memory-augmented, checkpointable coding agent."""


@main.command()
def init() -> None:
    """Set up .mazu/ (project memory) and git in the current directory."""
    root = Path.cwd()
    mazu_dir = _mazu_dir(root)
    already_existed = mazu_dir.exists()

    store = MemoryStore(_memory_db_path(root))
    store.close()
    ensure_gitignore(root)

    checkpoint_manager = CheckpointManager(root)
    was_git_repo = checkpoint_manager.is_git_repo()
    checkpoint_manager.ensure_git_repo()

    if already_existed:
        click.echo(f"Mazu already initialized at {mazu_dir}")
    else:
        click.echo(f"Initialized Mazu project memory at {mazu_dir}")
    if not was_git_repo:
        click.echo("Initialized a git repository (needed for checkpoints).")


_STATUS_MARK = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}


@main.command()
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Also make one minimal real API call per configured provider to confirm the "
    "key actually authenticates (costs a fraction of a cent per provider checked).",
)
@click.option(
    "--fix",
    is_flag=True,
    default=False,
    help="Automatically fix what's safely fixable without a value only you have: "
    "adds .mazu/ to .gitignore if missing, initializes a git repo if this directory "
    "isn't one yet. Doesn't touch API keys, Python version, or package installs -- "
    "use `mazu setup` for API keys.",
)
def doctor(live: bool, fix: bool) -> None:
    """Diagnose common setup problems: Python/git availability, which provider keys
    are configured, and whether the current directory is ready for `mazu run`."""
    root = Path.cwd()

    if fix:
        fixed = apply_fixes(root)
        if fixed:
            for description in fixed:
                click.echo(f"[fix] {description}")
            click.echo()
        else:
            click.echo("[fix] Nothing to fix.\n")

    results = run_diagnostics(root, live=live)

    for r in results:
        click.echo(f"{_STATUS_MARK[r.status]} {r.name}: {r.message}")

    fail_count = sum(1 for r in results if r.status == "fail")
    warn_count = sum(1 for r in results if r.status == "warn")
    click.echo()
    if fail_count:
        click.echo(f"{fail_count} problem(s) found that will likely block mazu from working.")
    elif warn_count:
        click.echo(f"No blocking problems, but {warn_count} thing(s) worth a look.")
    else:
        click.echo("Everything looks good.")


_SETUP_PROVIDER_CHOICES = ["anthropic", "openai", "deepseek", "gemini"]


@main.command("setup")
def setup_wizard() -> None:
    """Interactive first-run wizard: pick a provider, paste an API key, verify it
    works, and initialize the current directory. Everything it does is also doable
    piece by piece via `mazu config set` / `mazu init` / `mazu doctor` -- this just
    walks through them in one guided pass for a first-time setup."""
    from mazu.llm.client import _PROVIDER_DEFAULT_MODELS, _PROVIDERS

    click.echo("Mazu setup -- let's get you connected to a model provider.\n")
    provider_name = click.prompt(
        "Which provider do you want to use?",
        type=click.Choice(_SETUP_PROVIDER_CHOICES),
        default="anthropic",
    )
    provider = _PROVIDERS[provider_name]
    env_var = provider.api_key_env

    key = click.prompt(f"Paste your {env_var}", hide_input=True)
    config_key = f"{provider_name}_api_key"
    set_config_value(config_key, key)
    click.echo(f"Saved to {config_path()} (masked in `mazu config list`).\n")

    if click.confirm("Verify this key works with a real API call now?", default=True):
        import os

        # Verifying in-process needs the key in the environment right now --
        # set_config_value() only persists it to disk; env vars aren't re-read from
        # config.toml until the next command's load_config() call.
        os.environ[env_var] = key
        model = _PROVIDER_DEFAULT_MODELS[provider_name]
        result = check_live_api_key(provider_name, model)
        click.echo(f"[{result.status.upper()}] {result.message}")
        if result.status != "ok":
            click.echo(
                "The key is still saved -- fix and re-run `mazu setup`, or "
                "`mazu config set` directly, whenever you're ready."
            )
        click.echo()

    default_model_choice = _PROVIDER_DEFAULT_MODELS[provider_name]
    if click.confirm(f"Set {default_model_choice} as your default model?", default=True):
        set_config_value("default_model", default_model_choice)
        click.echo(f"default_model set to {default_model_choice}.\n")

    root = Path.cwd()
    if not (root / ".mazu").exists():
        if click.confirm(f"Initialize Mazu in the current directory ({root})?", default=True):
            click.get_current_context().invoke(init)

    click.echo('\nSetup complete. Try `mazu chat` or `mazu run "..."` to get started.')


@main.command("models")
def models_cmd() -> None:
    """Show what Mazu knows about each provider/model: real streaming support, tool
    use, context window, and approximate pricing. Best-effort and may go stale --
    verify against your provider's own docs before relying on it for capacity planning."""
    rows = list_capabilities()
    click.echo(f"{'MODEL':<32} {'STREAM':<7} {'TOOLS':<6} {'CONTEXT':<10} {'$/1M IN':<9} {'$/1M OUT'}")
    for r in rows:
        key = f"{r.provider}:{r.model}"
        stream = "yes" if r.streaming else "no"
        tools = "yes" if r.tool_use else "no"
        ctx = f"{r.context_window:,}" if r.context_window is not None else "?"
        price_in = f"${r.input_price_per_million:.2f}" if r.input_price_per_million is not None else "?"
        price_out = f"${r.output_price_per_million:.2f}" if r.output_price_per_million is not None else "?"
        click.echo(f"{key:<32} {stream:<7} {tools:<6} {ctx:<10} {price_in:<9} {price_out}")
    click.echo(
        "\nContext windows and pricing are best-effort (see mazu/llm/capabilities.py, "
        "mazu/llm/pricing.py) and may be stale -- treat as approximate."
    )


def _mask_secret(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


@main.group("config")
def config_group() -> None:
    """Manage persistent settings (~/.mazu/config.toml): default model, per-provider
    API keys. Env vars always take priority over anything set here."""


@config_group.command("list")
def config_list() -> None:
    """Show every config value currently set. API keys are masked."""
    values = list_config()
    if not values:
        click.echo(f"No config set. ({config_path()} doesn't exist or is empty.)")
        return
    for key, value in sorted(values.items()):
        display = _mask_secret(value) if key in _SECRET_CONFIG_KEYS else value
        click.echo(f"{key} = {display}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a config value, e.g. `mazu config set default_model anthropic:claude-opus-4-8`."""
    try:
        set_config_value(key, value)
    except ValueError as e:
        raise click.UsageError(str(e))
    display = _mask_secret(value) if key in _SECRET_CONFIG_KEYS else value
    click.echo(f"Set {key} = {display}")


@config_group.command("unset")
@click.argument("key")
def config_unset(key: str) -> None:
    """Remove a config value."""
    ok = unset_config_value(key)
    click.echo(f"Unset {key}." if ok else f"{key} was not set.")


@main.command()
@click.option(
    "--model",
    default=None,
    help="Override the model, e.g. 'deepseek:deepseek-chat' (default: env MAZU_MODEL, "
    "or auto-detected from whichever provider's API key is set).",
)
@click.option(
    "--shell-allowlist",
    default=None,
    help="Comma-separated program names (e.g. 'git,npm,pytest') -- if set, shell "
    "commands are only allowed if they start with one of these. The safety denylist "
    "still applies on top regardless. Default: no allowlist, every non-denylisted "
    "command is allowed (unchanged behavior).",
)
def chat(model: str | None, shell_allowlist: str | None) -> None:
    """Start an interactive chat session in the current directory."""
    ensure_api_key(model)
    root = Path.cwd()
    ensure_gitignore(root)

    memory_store = MemoryStore(_memory_db_path(root))
    global_memory_store = MemoryStore(_global_memory_db_path())
    skill_manager = SkillManager(root)
    checkpoint_manager = CheckpointManager(root)
    usage_store = UsageStore(_usage_db_path())
    action_log_store = ActionLogStore(_action_log_db_path(root))
    session_id = str(uuid.uuid4())

    registry = _build_registry(root, memory_store, global_memory_store, skill_manager, session_id)

    run_chat_loop(
        registry,
        session_id=session_id,
        memory_store=memory_store,
        global_memory_store=global_memory_store,
        skill_manager=skill_manager,
        checkpoint_manager=checkpoint_manager,
        model=model,
        usage_store=usage_store,
        action_log_store=action_log_store,
        shell_allowlist=_parse_shell_allowlist(shell_allowlist),
    )


@main.command()
@click.argument("task", required=False)
@click.option("--max-steps", default=15, show_default=True, help="Stop after this many tool-use rounds.")
@click.option(
    "--checkpoint-every", default=1, show_default=True, help="Snapshot every N rounds."
)
@click.option(
    "--allow-shell",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt for shell commands (a hardcoded safety denylist still applies).",
)
@click.option(
    "--keep-checkpoints",
    default=None,
    type=int,
    help="Keep this many most-recent checkpoints on disk, pruning older ones automatically (default: 50).",
)
@click.option(
    "--max-cost",
    default=None,
    type=float,
    help="Stop once the estimated spend (approximate, based on a built-in pricing table) "
    "reaches this many USD. Ignored with a warning if the model has no pricing data. "
    "Checked after each step completes, so one step's cost can land after the limit is "
    "reached before the run stops.",
)
@click.option(
    "--shell-allowlist",
    default=None,
    help="Comma-separated program names (e.g. 'git,npm,pytest') -- if set, shell "
    "commands are only allowed if they start with one of these. The safety denylist "
    "still applies on top regardless. Default: no allowlist, every non-denylisted "
    "command is allowed (unchanged behavior).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what the task would do -- which files would be written/edited and which "
    "shell commands would run -- without touching disk, executing anything, or creating "
    "checkpoints. Read-only tools (read_file, list_dir, glob_files, recall) still run for "
    "real so the model can gather real information while planning.",
)
@click.option(
    "--resume",
    "resume_run_id",
    default=None,
    help="Resume an earlier run by its id (shown in the end-of-run report, or via `mazu "
    "runs`), continuing from its last checkpoint's conversation state. Reuses that run's "
    "original task/model/options exactly -- other flags on this invocation are ignored. "
    "Mutually exclusive with passing a new TASK.",
)
@click.option(
    "--from-checkpoint",
    "from_checkpoint_id",
    default=None,
    help="Fork a new, divergent line of execution from an earlier checkpoint's state "
    "(git commit + memory.db + skills), then run TASK on it -- unlike --resume, this is "
    "NOT a continuation of the original run: it gets its own new run id on a new git "
    "branch, and the original branch's later history is untouched. Requires --branch. "
    "Mutually exclusive with --resume.",
)
@click.option(
    "--branch",
    "fork_branch_name",
    default=None,
    help="Name for the new git branch created by --from-checkpoint. Required together "
    "with --from-checkpoint.",
)
@click.option("--model", default=None, help="Override the model.")
def run(
    task: str | None,
    max_steps: int,
    checkpoint_every: int,
    allow_shell: bool,
    keep_checkpoints: int | None,
    max_cost: float | None,
    shell_allowlist: str | None,
    dry_run: bool,
    resume_run_id: str | None,
    from_checkpoint_id: str | None,
    fork_branch_name: str | None,
    model: str | None,
) -> None:
    """Run a task autonomously (multi-step, unattended), checkpointing along the way."""
    root = Path.cwd()
    ensure_gitignore(root)
    checkpoint_kwargs = {"retention": keep_checkpoints} if keep_checkpoints is not None else {}
    checkpoint_manager = CheckpointManager(root, **checkpoint_kwargs)
    run_store = RunStore(_runs_db_path(root))

    if from_checkpoint_id is not None and resume_run_id is not None:
        run_store.close()
        raise click.UsageError("Pass either --from-checkpoint or --resume, not both.")
    if fork_branch_name is not None and from_checkpoint_id is None:
        run_store.close()
        raise click.UsageError("--branch requires --from-checkpoint.")

    resume_messages = None
    origin_checkpoint_id = None
    parent_run_id = None
    branch_name = None
    if from_checkpoint_id is not None:
        if not fork_branch_name:
            run_store.close()
            raise click.UsageError("--from-checkpoint requires --branch <new-branch-name>.")
        if task is None:
            run_store.close()
            raise click.UsageError("--from-checkpoint requires a TASK to run on the new branch.")
        try:
            origin_entry = checkpoint_manager.show_entry(from_checkpoint_id)
        except ValueError as e:
            run_store.close()
            click.echo(str(e))
            return
        try:
            fork_result = checkpoint_manager.fork(origin_entry["id"], fork_branch_name)
        except ValueError as e:
            run_store.close()
            click.echo(str(e))
            return
        origin_checkpoint_id = origin_entry["id"]
        parent_run_id = origin_entry.get("session_id")
        branch_name = fork_branch_name
        resume_messages = fork_result["messages"]
        click.echo(
            f"Forked from {origin_checkpoint_id} onto new branch {branch_name!r} "
            f"({len(resume_messages)} prior message(s)). Will run: {task}"
        )
    elif resume_run_id is not None:
        if task is not None:
            run_store.close()
            raise click.UsageError("Pass either a new TASK or --resume <run_id>, not both.")
        run_row = run_store.get(resume_run_id)
        if run_row is None:
            run_store.close()
            click.echo(f"No run found with id {resume_run_id}.")
            return
        checkpoint_entry = checkpoint_manager.latest_for_session(resume_run_id)
        if checkpoint_entry is None:
            run_store.close()
            click.echo(
                f"No checkpoint found for run {resume_run_id} -- nothing to resume from "
                "(a dry run never checkpoints, and a run that failed before its first "
                "checkpoint-every boundary has no saved state either)."
            )
            return
        resume_messages = checkpoint_manager.inspect_conversation(checkpoint_entry["id"])
        task = run_row["task"]
        model = run_row["model"]
        max_steps = run_row["max_steps"]
        checkpoint_every = run_row["checkpoint_every"]
        allow_shell = bool(run_row["allow_shell"])
        shell_allowlist = run_row["shell_allowlist"]
        max_cost = run_row["max_cost"]
        dry_run = bool(run_row["dry_run"])
        click.echo(
            f"Resuming run {resume_run_id} from {checkpoint_entry['id']} "
            f"({len(resume_messages)} prior message(s)). Using this run's original config: "
            f"model={model} max-steps={max_steps} checkpoint-every={checkpoint_every} "
            f"allow-shell={allow_shell} dry-run={dry_run}."
        )
    elif task is None:
        run_store.close()
        raise click.UsageError("Provide a TASK, or use --resume <run_id> to continue an earlier run.")

    ensure_api_key(model)

    memory_store = MemoryStore(_memory_db_path(root))
    global_memory_store = MemoryStore(_global_memory_db_path())
    skill_manager = SkillManager(root)
    usage_store = UsageStore(_usage_db_path())
    action_log_store = ActionLogStore(_action_log_db_path(root))
    session_id = resume_run_id if resume_run_id is not None else str(uuid.uuid4())

    registry = _build_registry(
        root, memory_store, global_memory_store, skill_manager, session_id, dry_run=dry_run
    )

    run_autonomous(
        registry,
        task=task,
        session_id=session_id,
        checkpoint_manager=checkpoint_manager,
        memory_store=memory_store,
        global_memory_store=global_memory_store,
        skill_manager=skill_manager,
        max_steps=max_steps,
        checkpoint_every=checkpoint_every,
        allow_shell=allow_shell,
        max_cost=max_cost,
        model=model,
        usage_store=usage_store,
        action_log_store=action_log_store,
        shell_allowlist=_parse_shell_allowlist(shell_allowlist),
        dry_run=dry_run,
        run_store=run_store,
        resume_messages=resume_messages,
        origin_checkpoint_id=origin_checkpoint_id,
        parent_run_id=parent_run_id,
        branch_name=branch_name,
    )


@main.command("runs")
@click.option("--limit", default=20, show_default=True, type=int)
def runs_cmd(limit: int) -> None:
    """List recent `mazu run` invocations: id, status, stop reason, step progress."""
    root = Path.cwd()
    store = RunStore(_runs_db_path(root))
    rows = store.list_runs(limit=limit)
    store.close()
    if not rows:
        click.echo("No runs recorded yet.")
        return
    for r in rows:
        dry_marker = " [dry-run]" if r["dry_run"] else ""
        click.echo(
            f"{r['id']}  [{r['status']}]{dry_marker}  stop: {r['stop_reason'] or '-'}  "
            f"step {r['last_step']}/{r['max_steps']}  checkpoints: {r['checkpoints_created']}  "
            f"{r['started_at']}"
        )


@main.command("ui")
def ui_cmd() -> None:
    """Launch the terminal UI: browse checkpoints, memory, and the action log
    interactively. Requires the current directory to already be a Mazu project
    (`mazu init` first) and the optional `mazu[ui]` extra."""
    root = Path.cwd()
    if not (root / ".mazu").exists():
        click.echo("No .mazu/ here yet -- run `mazu init` first.")
        return
    try:
        from mazu.ui.app import MazuApp
    except ImportError:
        click.echo(
            'The terminal UI needs the "ui" extra. Install it with: pip install "mazu[ui]"'
        )
        return
    MazuApp(root).run()


DEFAULT_COUNCIL_MODELS = "anthropic:claude-sonnet-5,anthropic:claude-opus-4-8"
DEFAULT_COUNCIL_LEAD = "anthropic:claude-opus-4-8"


@main.command()
@click.argument("question")
@click.option(
    "--models",
    default=DEFAULT_COUNCIL_MODELS,
    show_default=True,
    help="Comma-separated provider:model list to ask independently, e.g. "
    "'anthropic:claude-sonnet-5,openai:gpt-5'.",
)
@click.option(
    "--lead",
    default=DEFAULT_COUNCIL_LEAD,
    show_default=True,
    help="Model that reviews all answers and gives the final recommendation.",
)
@click.option(
    "--max-cost",
    default=None,
    type=float,
    help="Stop once the estimated spend (approximate, based on a built-in pricing table) "
    "reaches this many USD across all council members combined. Ignored with a warning "
    "if none of the models have pricing data. Checked after each round completes, so one "
    "round's cost can land after the limit is reached before members stop; the lead "
    "synthesis call is skipped entirely if the budget is already exhausted.",
)
def council(question: str, models: str, lead: str, max_cost: float | None) -> None:
    """Ask multiple models the same question (read-only, advisory) and have a lead model
    pick the best answer. Costs one API call per model plus one for the lead — an opt-in,
    higher-cost mode for decisions worth a second (and third) opinion, not the default flow.
    """
    ensure_api_key(lead)  # members that lack a configured key fail individually and are reported, not fatal
    print_banner()
    root = Path.cwd()
    ensure_gitignore(root)

    memory_store = MemoryStore(_memory_db_path(root))
    global_memory_store = MemoryStore(_global_memory_db_path())
    skill_manager = SkillManager(root)
    usage_store = UsageStore(_usage_db_path())
    action_log_store = ActionLogStore(_action_log_db_path(root))
    session_id = str(uuid.uuid4())
    registry = _build_registry(root, memory_store, global_memory_store, skill_manager, session_id)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    try:
        run_council(
            question,
            models=model_list,
            lead_model=lead,
            full_registry=registry,
            memory_store=memory_store,
            global_memory_store=global_memory_store,
            skill_manager=skill_manager,
            usage_store=usage_store,
            session_id=session_id,
            action_log_store=action_log_store,
            max_cost=max_cost,
        )
    finally:
        memory_store.close()
        global_memory_store.close()
        usage_store.close()
        action_log_store.close()


@main.group(invoke_without_command=True)
@click.pass_context
def checkpoint(ctx: click.Context) -> None:
    """Manually snapshot the current code + memory state, or manage checkpoints.
    Run with no subcommand to take a snapshot (no live conversation to attach)."""
    if ctx.invoked_subcommand is not None:
        return
    root = Path.cwd()
    ensure_gitignore(root)
    checkpoint_manager = CheckpointManager(root)
    entry = checkpoint_manager.snapshot(messages=[], trigger="manual_cli")
    click.echo(f"Checkpoint {entry['id']} saved (commit {entry['git_commit'][:8]}).")


@checkpoint.command("list")
def checkpoint_list() -> None:
    """List all checkpoints for the current project, most recent last."""
    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    entries = checkpoint_manager.list_checkpoints()
    if not entries:
        click.echo("No checkpoints yet.")
        return
    for entry in entries:
        click.echo(
            f"{entry['id']}  {entry['created_at']}  ({entry['trigger']}) {entry['summary']}"
        )


@checkpoint.command("show")
@click.argument("checkpoint_id", required=False)
def checkpoint_show(checkpoint_id: str | None) -> None:
    """Show one checkpoint's full detail (defaults to the most recent one)."""
    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    try:
        entry = checkpoint_manager.show_entry(checkpoint_id)
    except ValueError as e:
        click.echo(str(e))
        return
    click.echo(f"Checkpoint {entry['id']}")
    click.echo(f"  Created:          {entry['created_at']}")
    click.echo(f"  Trigger:          {entry['trigger']}")
    click.echo(f"  Summary:          {entry['summary']}")
    click.echo(f"  Git commit:       {entry['git_commit'][:8]}")
    click.echo(f"  Conversation:     {entry['message_count']} message(s)")
    click.echo(f"  Memory snapshot:  {'yes' if entry['has_memory_snapshot'] else 'no'}")
    click.echo(f"  Skills snapshot:  {'yes' if entry['has_skills_snapshot'] else 'no'}")


@checkpoint.command("diff")
@click.argument("checkpoint_id", required=False)
def checkpoint_diff(checkpoint_id: str | None) -> None:
    """Show what's changed between a checkpoint and the current working tree
    (defaults to the most recent checkpoint). Read-only -- does not touch anything,
    unlike `mazu rollback`."""
    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    try:
        entry, diff = checkpoint_manager.diff_against_current(checkpoint_id)
    except ValueError as e:
        click.echo(str(e))
        return
    click.echo(f"Diff since {entry['id']} ({entry['created_at']}):\n")
    if diff.strip():
        click.echo(diff)
    else:
        click.echo("(no changes)")


@checkpoint.command("inspect")
@click.argument("checkpoint_id", required=False)
@click.option("--memory", "show_memory", is_flag=True, default=False, help="Show the memory captured in this checkpoint's snapshot.")
@click.option("--conversation", "show_conversation", is_flag=True, default=False, help="Show the conversation transcript captured in this checkpoint.")
def checkpoint_inspect(checkpoint_id: str | None, show_memory: bool, show_conversation: bool) -> None:
    """Show the actual content of a checkpoint's snapshot -- not just its metadata
    (see `checkpoint show` for that). Reads straight from the frozen snapshot, not
    the live/current state, so this reflects exactly what things looked like at
    that point in history."""
    if not show_memory and not show_conversation:
        click.echo("Pass --memory and/or --conversation to choose what to show.")
        return
    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    try:
        entry = checkpoint_manager.show_entry(checkpoint_id)
    except ValueError as e:
        click.echo(str(e))
        return

    click.echo(f"Checkpoint {entry['id']} ({entry['created_at']})\n")

    if show_memory:
        memories = checkpoint_manager.inspect_memory(entry["id"])
        click.echo(f"Memory snapshot ({len(memories)} active row(s)):")
        if not memories:
            click.echo("  (none captured)")
        for m in memories:
            click.echo(f"  [{m['id']}] ({m['category']}) {m['title']}: {m['body']}")
        click.echo()

    if show_conversation:
        messages = checkpoint_manager.inspect_conversation(entry["id"])
        click.echo(f"Conversation snapshot ({len(messages)} message(s)):")
        if not messages:
            click.echo("  (none captured)")
        for msg in messages:
            content = msg["content"]
            text = content if isinstance(content, str) else str(content)
            preview = text if len(text) <= 200 else text[:200] + "..."
            click.echo(f"  [{msg['role']}] {preview}")


@checkpoint.command("compare")
@click.argument("checkpoint_id_a")
@click.argument("checkpoint_id_b")
def checkpoint_compare(checkpoint_id_a: str, checkpoint_id_b: str) -> None:
    """Diff between two checkpoints directly, not either one vs. the current
    working tree (see `checkpoint diff` for that)."""
    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    try:
        entry_a, entry_b, diff = checkpoint_manager.compare(checkpoint_id_a, checkpoint_id_b)
    except ValueError as e:
        click.echo(str(e))
        return
    click.echo(f"Diff from {entry_a['id']} ({entry_a['created_at']}) to {entry_b['id']} ({entry_b['created_at']}):\n")
    click.echo(diff if diff.strip() else "(no changes)")


def _common_ancestor_id(
    checkpoint_manager: CheckpointManager, checkpoint_id_a: str, checkpoint_id_b: str, max_hops: int = 20
) -> str | None:
    """Walks each checkpoint's parent_checkpoint_id chain looking for the first id
    shared by both -- purely a readability aid for `compare-branches`, recomputed on
    demand from data already in index.json rather than tracked as its own stored
    field. Capped at max_hops so a corrupted/cyclic parent chain (should never
    happen, but this is display code, not something that should be able to hang)
    can't cause an unbounded walk.
    """
    by_id = {e["id"]: e for e in checkpoint_manager.list_checkpoints()}

    def ancestors(start_id: str) -> list[str]:
        chain = []
        current_id = start_id
        hops = 0
        while current_id is not None and current_id in by_id and hops < max_hops:
            chain.append(current_id)
            current_id = by_id[current_id].get("parent_checkpoint_id")
            hops += 1
        return chain

    chain_a = ancestors(checkpoint_id_a)
    chain_b = set(ancestors(checkpoint_id_b))
    return next((cid for cid in chain_a if cid in chain_b), None)


@checkpoint.command("compare-branches")
@click.argument("run_id_a")
@click.argument("run_id_b")
def checkpoint_compare_branches(run_id_a: str, run_id_b: str) -> None:
    """Compare two runs' outcomes side by side -- status, steps, estimated cost,
    checkpoints, memories saved -- plus a diff between their final checkpointed
    states. Meant for comparing sibling branches forked from a shared ancestor
    checkpoint (`mazu run --from-checkpoint`), but works for any two run ids."""
    root = Path.cwd()
    run_store = RunStore(_runs_db_path(root))
    row_a = run_store.get(run_id_a)
    row_b = run_store.get(run_id_b)
    if row_a is None or row_b is None:
        missing = run_id_a if row_a is None else run_id_b
        run_store.close()
        click.echo(f"No run found with id {missing}.")
        return
    run_store.close()

    usage_store = UsageStore(_usage_db_path())
    cost_a = usage_store.summary(session_id=run_id_a)["total_cost"]
    cost_b = usage_store.summary(session_id=run_id_b)["total_cost"]
    usage_store.close()

    def _print_run(label: str, row, cost: float) -> None:
        click.echo(f"{label}: {row['id']}")
        click.echo(f"  Branch:            {row['branch_name'] or '(default)'}")
        click.echo(f"  Status:            {row['status']}")
        click.echo(f"  Stop reason:       {row['stop_reason'] or 'unknown'}")
        click.echo(f"  Steps:             {row['last_step']}/{row['max_steps']}")
        click.echo(f"  Checkpoints:       {row['checkpoints_created']}")
        click.echo(f"  Memories saved:    {row['memories_saved']}")
        click.echo(f"  Estimated cost:    ~${cost:.4f}")
        click.echo(f"  Final checkpoint:  {row['last_checkpoint_id'] or '(none)'}")

    _print_run("Run A", row_a, cost_a)
    click.echo()
    _print_run("Run B", row_b, cost_b)
    click.echo()

    if not row_a["last_checkpoint_id"] or not row_b["last_checkpoint_id"]:
        click.echo("At least one run has no checkpoints -- nothing to diff.")
        return

    checkpoint_manager = CheckpointManager(root)
    try:
        entry_a, entry_b, diff = checkpoint_manager.compare(
            row_a["last_checkpoint_id"], row_b["last_checkpoint_id"]
        )
    except ValueError as e:
        click.echo(str(e))
        return

    ancestor_id = _common_ancestor_id(checkpoint_manager, entry_a["id"], entry_b["id"])
    if ancestor_id is not None:
        click.echo(f"Common ancestor: {ancestor_id}\n")

    click.echo(f"Diff between final states ({entry_a['id']} vs {entry_b['id']}):\n")
    click.echo(diff if diff.strip() else "(no changes)")


@checkpoint.command("prune")
@click.option(
    "--keep",
    default=None,
    type=int,
    help="Keep this many most-recent checkpoints (default: 50). Git history is untouched — "
    "this only removes the redundant memory.db/skills/conversation.json copies.",
)
def checkpoint_prune(keep: int | None) -> None:
    """Delete on-disk snapshot data for checkpoints beyond the most recent N."""
    root = Path.cwd()
    checkpoint_kwargs = {"retention": keep} if keep is not None else {}
    checkpoint_manager = CheckpointManager(root, **checkpoint_kwargs)
    pruned = checkpoint_manager.prune()
    click.echo(f"Pruned {pruned} checkpoint(s)." if pruned else "Nothing to prune.")


@main.command("timeline")
def timeline() -> None:
    """Readable history of every checkpoint: what changed since the previous one,
    and whether a memory/skills snapshot exists — a step-by-step view, not just a
    flat id list (see `mazu checkpoint list` for that)."""
    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    entries = checkpoint_manager.timeline_entries()
    if not entries:
        click.echo("No checkpoints yet.")
        return
    for entry in entries:
        click.echo(f"{entry['id']}  {entry['created_at']}  ({entry['trigger']}) {entry['summary']}")
        click.echo(f"    commit: {entry['git_commit'][:8]}")
        if entry["files_changed"]:
            click.echo(f"    files:  {', '.join(entry['files_changed'])}")
        else:
            click.echo("    files:  (first checkpoint — nothing to compare against)")
        flags = []
        flags.append("memory" if entry["has_memory_snapshot"] else "no memory")
        flags.append("skills" if entry["has_skills_snapshot"] else "no skills")
        click.echo(f"    snapshot: {', '.join(flags)}")
        click.echo()


@main.command("branch-from")
@click.argument("checkpoint_id")
@click.argument("branch_name")
def branch_from(checkpoint_id: str, branch_name: str) -> None:
    """Create a new git branch pointing at a checkpoint's commit, without rolling
    back or touching your current branch/working tree. Git-only -- memory and
    skills stay as they currently are; use `mazu rollback` instead if you need
    those restored too. Switch to the new branch yourself when ready: `git
    checkout <branch_name>`."""
    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    try:
        entry = checkpoint_manager.branch_from(checkpoint_id, branch_name)
    except ValueError as e:
        click.echo(str(e))
        return
    click.echo(f"Created branch '{branch_name}' at {entry['id']} (commit {entry['git_commit'][:8]}).")
    click.echo(f"Your current branch is unchanged. Switch with: git checkout {branch_name}")


@main.command("usage")
@click.option(
    "--since-days",
    default=None,
    type=int,
    help="Only include the last N days (default: all time).",
)
def usage_cmd(since_days: int | None) -> None:
    """Show estimated API spend across every mazu session (all projects, all
    providers) — approximate, based on the same built-in pricing table --max-cost
    uses, not a real billing figure."""
    store = UsageStore(_usage_db_path())
    summary = store.summary(since_days=since_days)
    store.close()

    if summary["total_calls"] == 0:
        click.echo("No usage recorded yet.")
        return

    window = f"last {since_days} day(s)" if since_days is not None else "all time"
    click.echo(f"Estimated spend ({window}): ${summary['total_cost']:.4f} across {summary['total_calls']} calls\n")
    click.echo("By model:")
    for row in summary["by_model"]:
        cost = f"${row['cost']:.4f}" if row["cost"] is not None else "(no pricing data)"
        click.echo(
            f"  {row['provider']}:{row['model']:<28} {row['calls']:>4} calls   "
            f"{row['input_tokens']:>8} in / {row['output_tokens']:>8} out   {cost}"
        )
    if summary["has_unpriced_calls"]:
        click.echo(
            "\nNote: some calls used a model with no entry in the pricing table and "
            "aren't reflected in the totals above."
        )


@main.group(invoke_without_command=True)
@click.pass_context
def log(ctx: click.Context) -> None:
    """Inspect the persistent agent action log: every tool call, across every
    session, in this project. With no subcommand, lists recent sessions."""
    if ctx.invoked_subcommand is None:
        _print_recent_sessions()


def _print_recent_sessions() -> None:
    store = ActionLogStore(_action_log_db_path(Path.cwd()))
    sessions = store.list_sessions()
    store.close()
    if not sessions:
        click.echo("No actions recorded yet.")
        return
    for s in sessions:
        click.echo(
            f"{s['session_id']}  ({s['command']})  {s['action_count']} action(s), "
            f"{s['error_count']} not-ok  {s['started_at']} -> {s['last_at']}"
        )


@log.command("show")
@click.argument("session_id")
def log_show(session_id: str) -> None:
    """Show the full action log for one session: every tool call, its input, outcome,
    and output, in order."""
    store = ActionLogStore(_action_log_db_path(Path.cwd()))
    actions = store.session_actions(session_id)
    store.close()
    if not actions:
        click.echo(f"No actions recorded for session {session_id}.")
        return
    for a in actions:
        click.echo(f"[{a['created_at']}] {a['tool_name']} — {a['outcome']}")
        click.echo(f"    input: {a['tool_input']}")
        click.echo(f"    output: {a['output_summary']}")
        if a["changed_file"]:
            click.echo(f"    changed: {a['changed_file']}")


@main.command()
@click.argument("checkpoint_id", required=False)
def rollback(checkpoint_id: str | None) -> None:
    """Roll back code + memory to a checkpoint (run outside an active `mazu chat` session)."""
    root = Path.cwd()
    checkpoint_manager = CheckpointManager(root)
    try:
        entry, diff_stat = checkpoint_manager.preview_rollback(checkpoint_id)
    except ValueError as e:
        click.echo(str(e))
        return
    click.echo(f"Rolling back to {entry['id']} ({entry['created_at']}): {entry['summary']}")
    if diff_stat.strip():
        click.echo(diff_stat)
    if not click.confirm("This discards code and memory changes since then. Proceed?"):
        click.echo("Rollback cancelled.")
        return
    checkpoint_manager.restore(entry["id"])
    click.echo(f"Restored to {entry['id']}. (Conversation state only restores inside a live `mazu chat` session.)")


@main.group()
def memory() -> None:
    """Inspect and manage project memory."""


@memory.command("list")
@click.option("--category", default=None, help="Filter by category.")
@click.option(
    "--global",
    "use_global",
    is_flag=True,
    default=False,
    help="List the global store (user_preference facts shared across every project) instead of this project's.",
)
def memory_list(category: str | None, use_global: bool) -> None:
    """List stored memories for the current project."""
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    rows = store.search(category=category, limit=200)
    store.close()
    if not rows:
        click.echo("No memories stored yet.")
        return
    for row in rows:
        pin = "*" if row["pinned"] else " "
        click.echo(
            f"[{row['id']:>4}]{pin} ({row['category']}, {row['source']}) {row['title']}"
        )
        click.echo(f"       {row['body']}")
        if row["tags"]:
            click.echo(f"       tags: {row['tags']}")
        retrieval_count = row["retrieval_count"] if "retrieval_count" in row.keys() else 0
        if retrieval_count:
            click.echo(f"       used {retrieval_count}x, last: {row['last_used_at']}")


@memory.command("forget")
@click.argument("memory_id", type=int)
@click.option(
    "--global",
    "use_global",
    is_flag=True,
    default=False,
    help="Forget from the global store instead of this project's.",
)
def memory_forget(memory_id: int, use_global: bool) -> None:
    """Delete a memory by id."""
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    ok = store.forget(memory_id)
    store.close()
    if ok:
        click.echo(f"Forgot memory {memory_id}.")
    else:
        click.echo(f"No memory with id {memory_id}.")


@memory.command("consolidate")
@click.option(
    "--threshold",
    default=FUZZY_DUPLICATE_THRESHOLD,
    show_default=True,
    type=float,
    help="Similarity threshold (0-1) for treating two memories as duplicates of "
    "each other. Same scale as remember's own fuzzy dedup check.",
)
@click.option(
    "--global",
    "use_global",
    is_flag=True,
    default=False,
    help="Consolidate the global store instead of this project's.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be merged without actually changing anything.",
)
def memory_consolidate(threshold: float, use_global: bool, dry_run: bool) -> None:
    """Find near-duplicate memories (only within the same category) and merge each
    group down to its newest entry, marking the rest as superseded. Nothing is
    deleted -- superseded memories stay in the database, just excluded from
    context injection and mazu memory list. Entirely local; no API calls."""
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    clusters = find_duplicate_clusters(store, threshold=threshold)

    if not clusters:
        click.echo("No near-duplicate memories found.")
        store.close()
        return

    if dry_run:
        click.echo(f"Would merge {len(clusters)} group(s) of near-duplicates:\n")
        for cluster in clusters:
            newest = max(cluster, key=lambda r: r["created_at"])
            others = [r for r in cluster if r["id"] != newest["id"]]
            click.echo(f"  Keep [{newest['id']}] {newest['title']} ({newest['category']})")
            for o in others:
                click.echo(f"    ← merges [{o['id']}] {o['title']}")
        click.echo("\n(dry run — nothing changed; re-run without --dry-run to apply)")
        store.close()
        return

    summary = apply_consolidation(store, clusters)
    store.close()
    click.echo(f"Merged {len(summary)} group(s) of near-duplicates:\n")
    for entry in summary:
        click.echo(f"  Kept [{entry['survivor_id']}] {entry['survivor_title']} ({entry['category']})")
        for s in entry["superseded"]:
            click.echo(f"    ← merged [{s['id']}] {s['title']}")


@memory.command("why")
@click.argument("query")
@click.option("--limit", default=15, show_default=True, type=int, help="Max ranked slots (matches build_context_block's default).")
@click.option(
    "--global",
    "use_global",
    is_flag=True,
    default=False,
    help="Explain retrieval against the global store instead of this project's.",
)
def memory_why(query: str, limit: int, use_global: bool) -> None:
    """Show which memories would be retrieved for QUERY and why (score, floor reason)."""
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    explanations = explain_retrieval(store, query=query, limit=limit)
    store.close()

    if not explanations:
        click.echo("No memories stored yet.")
        return

    for e in explanations:
        row = e["row"]
        mark = "x" if e["included"] else " "
        if e["combined"] is None:
            score_str = e["reason"]
        else:
            score_str = f"bm25={e['bm25']:.2f}"
            if e["semantic"] is not None:
                score_str += f" semantic={e['semantic']:.2f}"
            score_str += f" combined={e['combined']:.2f}"
        click.echo(f"[{mark}] [{row['id']:>4}] ({row['category']}) {row['title']} — {score_str}")


@memory.command("pin")
@click.argument("memory_id", type=int)
@click.option("--global", "use_global", is_flag=True, default=False, help="Pin in the global store instead of this project's.")
def memory_pin(memory_id: int, use_global: bool) -> None:
    """Pin a memory so it's always included in context, regardless of relevance ranking."""
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    ok = store.pin(memory_id)
    store.close()
    click.echo(f"Pinned memory {memory_id}." if ok else f"No memory with id {memory_id}.")


@memory.command("unpin")
@click.argument("memory_id", type=int)
@click.option("--global", "use_global", is_flag=True, default=False, help="Unpin in the global store instead of this project's.")
def memory_unpin(memory_id: int, use_global: bool) -> None:
    """Unpin a memory, letting it be ranked normally again."""
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    ok = store.unpin(memory_id)
    store.close()
    click.echo(f"Unpinned memory {memory_id}." if ok else f"No memory with id {memory_id}.")


@memory.command("edit")
@click.argument("memory_id", type=int)
@click.option("--title", default=None, help="New title.")
@click.option("--body", default=None, help="New body.")
@click.option("--global", "use_global", is_flag=True, default=False, help="Edit in the global store instead of this project's.")
def memory_edit(memory_id: int, title: str | None, body: str | None, use_global: bool) -> None:
    """Edit a memory's title and/or body in place."""
    if title is None and body is None:
        raise click.UsageError("Provide --title and/or --body to edit.")
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    ok = store.edit(memory_id, title=title, body=body)
    store.close()
    click.echo(f"Updated memory {memory_id}." if ok else f"No memory with id {memory_id}.")


@memory.command("supersede")
@click.argument("old_id", type=int)
@click.argument("new_id", type=int)
@click.option("--global", "use_global", is_flag=True, default=False, help="Supersede within the global store instead of this project's.")
def memory_supersede(old_id: int, new_id: int, use_global: bool) -> None:
    """Mark OLD_ID as replaced by NEW_ID -- retires it from context/list without deleting it."""
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    if store.get(old_id) is None:
        store.close()
        click.echo(f"No memory with id {old_id}.")
        return
    if store.get(new_id) is None:
        store.close()
        click.echo(f"No memory with id {new_id}.")
        return
    ok = store.supersede(old_id, new_id)
    store.close()
    click.echo(f"Memory {old_id} marked as superseded by {new_id}." if ok else f"Failed to supersede {old_id}.")


@memory.command("stats")
@click.option("--global", "use_global", is_flag=True, default=False, help="Show stats for the global store instead of this project's.")
def memory_stats(use_global: bool) -> None:
    """Summarize memory counts by category/source, plus oldest/newest and superseded count."""
    db_path = _global_memory_db_path() if use_global else _memory_db_path(Path.cwd())
    store = MemoryStore(db_path)
    stats = store.stats()
    store.close()

    click.echo(
        f"Total: {stats['total']} ({stats['active']} active, "
        f"{stats['superseded']} superseded, {stats['pinned']} pinned)"
    )
    if stats["by_category"]:
        click.echo("\nBy category:")
        for category, count in sorted(stats["by_category"].items()):
            click.echo(f"  {category}: {count}")
    if stats["by_source"]:
        click.echo("\nBy source:")
        for source, count in sorted(stats["by_source"].items()):
            click.echo(f"  {source}: {count}")
    if stats["oldest"]:
        click.echo(f"\nOldest: [{stats['oldest']['id']}] {stats['oldest']['title']} ({stats['oldest']['created_at']})")
    if stats["newest"]:
        click.echo(f"Newest: [{stats['newest']['id']}] {stats['newest']['title']} ({stats['newest']['created_at']})")


@main.group()
def skills() -> None:
    """Inspect and manage the local skill library."""


@skills.command("list")
def skills_list() -> None:
    """List saved skills for the current project."""
    root = Path.cwd()
    manager = SkillManager(root)
    metas = manager.list()
    if not metas:
        click.echo("No skills saved yet.")
        return
    for m in metas:
        click.echo(f"- {m['name']}: {m['description']} (used {m.get('usage_count', 0)}x)")


@skills.command("forget")
@click.argument("name")
def skills_forget(name: str) -> None:
    """Delete a saved skill by name."""
    root = Path.cwd()
    manager = SkillManager(root)
    if manager.delete(name):
        click.echo(f"Deleted skill '{name}'.")
    else:
        click.echo(f"No skill named '{name}'.")


if __name__ == "__main__":
    main()
