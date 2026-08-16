"""`mazu explore` — forks N branches from one checkpoint, runs the SAME task on
each with a DIFFERENT model in parallel (via real git worktrees, not sequential
checkouts on one working tree), optionally evaluates each branch with a test
command, and reports a ranked comparison. Strictly opt-in: never called from
`mazu run`/`mazu chat`, and always requires the user to explicitly invoke it.

Structurally parallel to mazu/agent/council.py (ThreadPoolExecutor fan-out, a
shared cost tracker, workers never touch a shared/global sqlite connection or
print directly) but runs full autonomous agents that actually write files and
create checkpoints, rather than council's read-only advisory Q&A.
"""

import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.agent.autonomous import run_autonomous
from mazu.agent.council import _SharedCostTracker
from mazu.agent.registry_factory import build_registry
from mazu.checkpoint.manager import CheckpointManager, _backup_sqlite
from mazu.checkpoint.worktree import add_worktree
from mazu.memory.store import MemoryStore
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.usage.store import UsageStore

# Hard cap, not user-configurable in v1 -- directly prevents a typo like
# `--approaches 50` from turning into an accidental large bill (N full autonomous
# agents running in parallel, each potentially many tool-call rounds).
MAX_EXPLORE_APPROACHES = 5

DEFAULT_MAX_EXPLORE_STEPS = 15

_TEST_TIMEOUT_SECONDS = 300
_TEST_OUTPUT_TAIL_LINES = 50


def _slugify_model(model: str, used: set[str]) -> str:
    """'anthropic:claude-sonnet-5' -> 'anthropic-claude-sonnet-5'. Appends -2, -3,
    ... on collision (the same model passed twice in --models) so two branches
    never compute the same worktree path/branch name.
    """
    base = model.lower()
    for ch in (":", "/", ".", "_", " "):
        base = base.replace(ch, "-")
    slug = base
    n = 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


def _restore_snapshot_into_worktree(checkpoint_manager: CheckpointManager, checkpoint_id: str, worktree_path: Path) -> None:
    """Copies a checkpoint's memory.db/skills snapshot into a freshly-created
    worktree's own .mazu/ -- the worktree-based analog of what
    CheckpointManager.fork() already does for its single working tree, since
    `git worktree add` only handles the tracked-file/git side, never touches
    the gitignored .mazu/ state.
    """
    checkpoint_dir = checkpoint_manager.checkpoints_dir / checkpoint_id
    dest_mazu = worktree_path / ".mazu"
    dest_mazu.mkdir(parents=True, exist_ok=True)

    snapshot_db = checkpoint_dir / "memory.db"
    if snapshot_db.exists():
        _backup_sqlite(snapshot_db, dest_mazu / "memory.db")

    snapshot_skills = checkpoint_dir / "skills"
    if snapshot_skills.exists():
        shutil.copytree(snapshot_skills, dest_mazu / "skills", dirs_exist_ok=True)


def _run_one_branch(
    model: str,
    branch_slug: str,
    worktree_path: Path,
    task: str,
    resume_messages: list[dict],
    origin_checkpoint_id: str,
    max_cost: float | None,
    shared_tracker: _SharedCostTracker,
    test_command: str | None,
    max_steps: int,
) -> dict:
    """Worker-thread body: one branch's full autonomous run plus (optionally) its
    test evaluation. Never prints or logs directly -- SQLite connections aren't
    safe to share across threads, and interleaved prints from N parallel branches
    are unreadable, same reasoning council.py's _ask_member already documents.
    Returns a plain dict; all display happens in the main thread after collection.
    """
    memory_db_path = worktree_path / ".mazu" / "memory.db"
    global_memory_db_path = Path.home() / ".mazu" / "global_memory.db"
    usage_db_path = Path.home() / ".mazu" / "usage.db"
    action_log_db_path = worktree_path / ".mazu" / "action_log.db"
    runs_db_path = worktree_path / ".mazu" / "runs.db"

    memory_store = MemoryStore(memory_db_path)
    global_memory_store = MemoryStore(global_memory_db_path)
    skill_manager = SkillManager(worktree_path)
    usage_store = UsageStore(usage_db_path)
    action_log_store = ActionLogStore(action_log_db_path)
    run_store = RunStore(runs_db_path)
    checkpoint_manager = CheckpointManager(worktree_path)
    session_id = str(uuid.uuid4())

    registry = build_registry(worktree_path, memory_store, global_memory_store, skill_manager, session_id)

    error: str | None = None
    try:
        run_autonomous(
            registry,
            task=task,
            session_id=session_id,
            checkpoint_manager=checkpoint_manager,
            memory_store=memory_store,
            global_memory_store=global_memory_store,
            skill_manager=skill_manager,
            max_steps=max_steps,
            # Forced on: an unattended parallel branch cannot answer an interactive
            # confirmation prompt (safe_confirm() blocks on input()) -- the hardcoded
            # shell denylist (mazu/tools/shell.py) still applies regardless, same
            # backstop --allow-shell already relies on in mazu run.
            allow_shell=True,
            max_cost=max_cost,
            model=model,
            usage_store=usage_store,
            action_log_store=action_log_store,
            run_store=run_store,
            resume_messages=resume_messages,
            origin_checkpoint_id=origin_checkpoint_id,
            branch_name=branch_slug,
            shared_cost_tracker=shared_tracker,
        )
        # run_autonomous's own `finally` block already closed run_store/usage_store/
        # action_log_store/global_memory_store -- re-open short-lived fresh
        # connections below to read the results back, same pattern
        # checkpoint_compare_branches already uses for a finished run's row.
    except Exception as e:  # noqa: BLE001 -- one branch's crash must not sink the others
        error = str(e)
    finally:
        memory_store.close()

    result_run_store = RunStore(runs_db_path)
    result_usage_store = UsageStore(usage_db_path)
    run_row = result_run_store.get(session_id)
    cost = result_usage_store.summary(session_id=session_id)["total_cost"]
    result_run_store.close()
    result_usage_store.close()

    test_passed: bool | None = None
    test_output_tail = ""
    if error is None and test_command:
        try:
            proc = subprocess.run(
                test_command,
                shell=True,
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=_TEST_TIMEOUT_SECONDS,
            )
            test_passed = proc.returncode == 0
            combined = (proc.stdout or "") + (proc.stderr or "")
            test_output_tail = "\n".join(combined.splitlines()[-_TEST_OUTPUT_TAIL_LINES:])
        except subprocess.TimeoutExpired:
            test_passed = False
            test_output_tail = f"(test command timed out after {_TEST_TIMEOUT_SECONDS}s)"

    return {
        "model": model,
        "branch_name": branch_slug,
        "session_id": session_id,
        "worktree_path": worktree_path,
        "run_row": dict(run_row) if run_row is not None else None,
        "cost": cost or 0.0,
        "test_passed": test_passed,
        "test_output_tail": test_output_tail,
        "error": error,
    }


def run_explore(
    task: str,
    models: list[str],
    root: Path,
    checkpoint_manager: CheckpointManager,
    from_checkpoint_id: str | None = None,
    max_cost: float | None = None,
    test_command: str | None = None,
    max_steps: int = DEFAULT_MAX_EXPLORE_STEPS,
) -> list[dict]:
    """Public entry point. Resolves the base checkpoint (most recent on the current
    branch if `from_checkpoint_id` is None, matching every other checkpoint-id
    default in this codebase), creates one worktree+branch per model, then fans
    the N autonomous runs out in parallel.
    """
    # Real bug caught in live testing: a project with no checkpoint yet (never ran
    # `mazu checkpoint`/`mazu run`) has an empty index.json, so
    # checkpoint_manager.show_entry(None) has nothing to resolve to and raises --
    # unlike `mazu run` (no --from-checkpoint), which never requires a pre-existing
    # checkpoint entry at all. explore always needs a real checkpoint id to fork N
    # branches from (each branch's first snapshot anchors its parent_checkpoint_id
    # to it), so auto-create one baseline checkpoint here rather than making the
    # user run a separate `mazu checkpoint` first just to unblock explore.
    if from_checkpoint_id is None and not checkpoint_manager.list_checkpoints():
        checkpoint_manager.snapshot(messages=[], trigger="explore_baseline")

    origin_entry = checkpoint_manager.show_entry(from_checkpoint_id)
    base_commit = origin_entry["git_commit"]
    resume_messages = checkpoint_manager.inspect_conversation(origin_entry["id"])

    group_id = uuid.uuid4().hex[:8]
    used_slugs: set[str] = set()
    branch_specs: list[tuple[str, str, Path]] = []
    for model in models:
        model_slug = _slugify_model(model, used_slugs)
        branch_slug = f"explore-{group_id}-{model_slug}"
        worktree_path = root / ".mazu" / "explore" / group_id / model_slug
        branch_specs.append((model, branch_slug, worktree_path))

    # Worktrees are created SEQUENTIALLY, before any threading starts -- git's own
    # worktree bookkeeping (.git/worktrees/) is not documented as safe for
    # concurrent `git worktree add` calls against one repo, unlike the parallel
    # AGENT work itself, which is safe once each branch has its own isolated
    # directory and .mazu/ state.
    for model, branch_slug, worktree_path in branch_specs:
        result = add_worktree(root, worktree_path, branch_slug, base_commit)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create a worktree for {model!r}: {result.stderr.strip() or result.stdout.strip()}"
            )
        _restore_snapshot_into_worktree(checkpoint_manager, origin_entry["id"], worktree_path)

    shared_tracker = _SharedCostTracker(max_cost)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(branch_specs)) as executor:
        future_to_model = {
            executor.submit(
                _run_one_branch,
                model,
                branch_slug,
                worktree_path,
                task,
                resume_messages,
                origin_entry["id"],
                max_cost,
                shared_tracker,
                test_command,
                max_steps,
            ): model
            for model, branch_slug, worktree_path in branch_specs
        }
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            result = future.result()
            print(f"[{model}] done")
            results.append(result)

    return results


def format_explore_report(results: list[dict], test_command: str | None) -> str:
    """The N-way generalization of `checkpoint compare-branches`'s existing
    (hardcoded-to-2) comparison display. Ranks by test-pass status first (only if
    `test_command` was actually given -- never fabricates a "winner" from a metric
    the user didn't ask for), then by estimated cost ascending. Never merges,
    checks out, or deletes anything -- every branch's adoption command is printed
    for the user to run themselves.
    """

    def sort_key(r: dict) -> tuple[int, float]:
        if test_command:
            passed = r.get("test_passed")
            pass_rank = 0 if passed is True else (1 if passed is None else 2)
        else:
            pass_rank = 0
        return (pass_rank, r.get("cost") or 0.0)

    ranked = sorted(results, key=sort_key)

    lines = ["=== Explore report ===", ""]
    for i, r in enumerate(ranked, 1):
        lines.append(f"{i}. {r['model']}  (branch: {r['branch_name']})")
        if r.get("error"):
            lines.append(f"   ERROR: {r['error']}")
            lines.append("")
            continue
        row = r.get("run_row") or {}
        lines.append(f"   Status:      {row.get('status', 'unknown')}")
        lines.append(f"   Stop reason: {row.get('stop_reason') or 'unknown'}")
        lines.append(f"   Steps:       {row.get('last_step', '?')}/{row.get('max_steps', '?')}")
        lines.append(f"   Checkpoints: {row.get('checkpoints_created', '?')}")
        lines.append(f"   Cost:        ~${r.get('cost', 0.0):.4f}")
        if test_command:
            passed = r.get("test_passed")
            status = "PASSED" if passed is True else ("FAILED" if passed is False else "not run")
            lines.append(f"   Tests:       {status}")
        # `git checkout <branch>` in the ORIGIN repo fails outright while its
        # worktree still exists (git refuses to have the same branch checked out
        # twice) -- a real thing caught in live testing, not a hypothetical. `cd`
        # into the worktree is always immediately usable; promoting the branch
        # back to the main repo requires releasing the worktree first.
        lines.append(f"   Adopt:       cd {r['worktree_path']}")
        lines.append(
            f"                (or, to check it out in this repo instead: "
            f"git worktree remove {r['worktree_path']} && git checkout {r['branch_name']})"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
