import sys
import uuid
from pathlib import Path

import click

from mazu import __version__
from mazu.agent.autonomous import run_autonomous
from mazu.agent.council import run_council
from mazu.agent.loop import run_chat_loop
from mazu.banner import print_banner
from mazu.checkpoint.manager import CheckpointManager
from mazu.config import ensure_api_key
from mazu.memory.store import MemoryStore
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

GITIGNORE_ENTRY = ".mazu/"


def _mazu_dir(root: Path) -> Path:
    return root / ".mazu"


def _memory_db_path(root: Path) -> Path:
    return _mazu_dir(root) / "memory.db"


def _global_memory_db_path() -> Path:
    # Deliberately not tied to any project root — this is the one store shared across
    # every project, for durable facts about the person, not the codebase.
    return Path.home() / ".mazu" / "global_memory.db"


def _usage_db_path() -> Path:
    # Global like global_memory.db — spend is tied to the person/API keys, not any
    # one project. A separate file on purpose (see UsageStore's docstring).
    return Path.home() / ".mazu" / "usage.db"


def _ensure_gitignore(root: Path) -> None:
    gitignore = root / ".gitignore"
    if gitignore.exists():
        if GITIGNORE_ENTRY not in gitignore.read_text(encoding="utf-8"):
            with open(gitignore, "a", encoding="utf-8") as f:
                f.write(f"\n{GITIGNORE_ENTRY}\n")
    else:
        gitignore.write_text(f"{GITIGNORE_ENTRY}\n", encoding="utf-8")


def _build_registry(
    root: Path,
    memory_store: MemoryStore,
    global_memory_store: MemoryStore,
    skill_manager: SkillManager,
    session_id: str,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in make_fs_tools(root):
        registry.register(tool)
    registry.register(make_shell_tool(root))
    for tool in make_memory_tools(memory_store, global_memory_store, session_id):
        registry.register(tool)
    for tool in make_skill_tools(skill_manager):
        registry.register(tool)
    return registry


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
    _ensure_gitignore(root)

    checkpoint_manager = CheckpointManager(root)
    was_git_repo = checkpoint_manager.is_git_repo()
    checkpoint_manager.ensure_git_repo()

    if already_existed:
        click.echo(f"Mazu already initialized at {mazu_dir}")
    else:
        click.echo(f"Initialized Mazu project memory at {mazu_dir}")
    if not was_git_repo:
        click.echo("Initialized a git repository (needed for checkpoints).")


@main.command()
@click.option(
    "--model",
    default=None,
    help="Override the model, e.g. 'deepseek:deepseek-chat' (default: env MAZU_MODEL, "
    "or auto-detected from whichever provider's API key is set).",
)
def chat(model: str | None) -> None:
    """Start an interactive chat session in the current directory."""
    ensure_api_key(model)
    root = Path.cwd()
    _ensure_gitignore(root)

    memory_store = MemoryStore(_memory_db_path(root))
    global_memory_store = MemoryStore(_global_memory_db_path())
    skill_manager = SkillManager(root)
    checkpoint_manager = CheckpointManager(root)
    usage_store = UsageStore(_usage_db_path())
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
    )


@main.command()
@click.argument("task")
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
    "reaches this many USD. Ignored with a warning if the model has no pricing data.",
)
@click.option("--model", default=None, help="Override the model.")
def run(
    task: str,
    max_steps: int,
    checkpoint_every: int,
    allow_shell: bool,
    keep_checkpoints: int | None,
    max_cost: float | None,
    model: str | None,
) -> None:
    """Run a task autonomously (multi-step, unattended), checkpointing along the way."""
    ensure_api_key(model)
    root = Path.cwd()
    _ensure_gitignore(root)

    memory_store = MemoryStore(_memory_db_path(root))
    global_memory_store = MemoryStore(_global_memory_db_path())
    skill_manager = SkillManager(root)
    checkpoint_kwargs = {"retention": keep_checkpoints} if keep_checkpoints is not None else {}
    checkpoint_manager = CheckpointManager(root, **checkpoint_kwargs)
    usage_store = UsageStore(_usage_db_path())
    session_id = str(uuid.uuid4())

    registry = _build_registry(root, memory_store, global_memory_store, skill_manager, session_id)

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
    )


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
def council(question: str, models: str, lead: str) -> None:
    """Ask multiple models the same question (read-only, advisory) and have a lead model
    pick the best answer. Costs one API call per model plus one for the lead — an opt-in,
    higher-cost mode for decisions worth a second (and third) opinion, not the default flow.
    """
    ensure_api_key(lead)  # members that lack a configured key fail individually and are reported, not fatal
    print_banner()
    root = Path.cwd()
    _ensure_gitignore(root)

    memory_store = MemoryStore(_memory_db_path(root))
    global_memory_store = MemoryStore(_global_memory_db_path())
    skill_manager = SkillManager(root)
    usage_store = UsageStore(_usage_db_path())
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
        )
    finally:
        memory_store.close()
        global_memory_store.close()
        usage_store.close()


@main.group(invoke_without_command=True)
@click.pass_context
def checkpoint(ctx: click.Context) -> None:
    """Manually snapshot the current code + memory state, or manage checkpoints.
    Run with no subcommand to take a snapshot (no live conversation to attach)."""
    if ctx.invoked_subcommand is not None:
        return
    root = Path.cwd()
    _ensure_gitignore(root)
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
