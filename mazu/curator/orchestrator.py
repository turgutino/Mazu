from dataclasses import dataclass, field
from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.lock import LockTimeoutError, ReentrantFileLock
from mazu.curator.areas import Area
from mazu.curator.areas.conflicts import CONFLICTS_AREA
from mazu.curator.areas.housekeeping import HOUSEKEEPING_AREA
from mazu.curator.areas.memory import MEMORY_AREA
from mazu.curator.areas.skills import SKILLS_AREA
from mazu.curator.budget import CuratorBudget
from mazu.curator.config import curator_configured, curator_enabled, curator_model
from mazu.curator.loop import run_curator_pass
from mazu.curator.prompts import CURATOR_SYSTEM_PROMPT
from mazu.curator.signals import is_eligible
from mazu.curator.store import CuratorStore, new_curator_run_id
from mazu.curator.tools.registry import build_curator_registry
from mazu.llm.client import _split_model
from mazu.llm.pricing import estimate_cost
from mazu.memory.store import MemoryStore
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.usage.store import UsageStore

# Every implemented area: memory, skills, housekeeping (runs/checkpoints/usage/
# config/council/action_log bundled into one, see areas/housekeeping.py), and
# conflicts (contradiction detection). The orchestrator loop below never changes
# to add a new area -- only this table and the area_name == "..." dispatch below
# that opens the right store(s) for build_curator_registry.
KNOWN_AREAS: dict[str, Area] = {
    "memory": MEMORY_AREA, "skills": SKILLS_AREA, "housekeeping": HOUSEKEEPING_AREA,
    "conflicts": CONFLICTS_AREA,
}


def _curator_db_path(root: Path) -> Path:
    return root / ".mazu" / "curator.db"


def _memory_db_path(root: Path) -> Path:
    return root / ".mazu" / "memory.db"


def _action_log_db_path(root: Path) -> Path:
    return root / ".mazu" / "action_log.db"


def _runs_db_path(root: Path) -> Path:
    return root / ".mazu" / "runs.db"


def _area_lock_path(root: Path, area_name: str) -> Path:
    return root / ".mazu" / f"curator_{area_name}.lock"


def _usage_db_path() -> Path:
    # Global like every other ~/.mazu/usage.db reader in the codebase -- see
    # UsageStore's own docstring for why this one store isn't project-scoped.
    return Path.home() / ".mazu" / "usage.db"


@dataclass
class AreaResult:
    area: str
    ran: bool
    final_text: str = ""
    cost: float = 0.0
    log_entries: int = 0
    skipped_reason: str | None = None


@dataclass
class CuratorRunSummary:
    ran: bool
    reason: str | None = None
    run_id: str | None = None
    areas: list[AreaResult] = field(default_factory=list)
    total_cost: float = 0.0


def run_curator(
    root: Path,
    areas: list[str] | None = None,
    full: bool = False,
    dry_run: bool = False,
    max_cost: float | None = None,
    max_rounds: int = 8,
    verbose: bool = False,
) -> CuratorRunSummary:
    """The one entry point `mazu curator run` calls. Guaranteed no-op (zero API
    calls, zero files written under .mazu/) whenever Curator isn't configured or
    has been disabled -- callers should treat CuratorRunSummary(ran=False) as
    "nothing happened, and that's expected," not an error.
    """
    if not curator_configured():
        return CuratorRunSummary(ran=False, reason="not_configured")
    if not curator_enabled():
        return CuratorRunSummary(ran=False, reason="disabled")

    requested = areas if areas else list(KNOWN_AREAS)
    unknown = [a for a in requested if a not in KNOWN_AREAS]
    if unknown:
        return CuratorRunSummary(ran=False, reason=f"unknown area(s): {', '.join(unknown)}")

    forced = bool(areas)  # an explicit --area X always runs, regardless of watermark/signal

    curator_store = CuratorStore(_curator_db_path(root))
    usage_store = UsageStore(_usage_db_path())
    run_id = new_curator_run_id()
    model = curator_model()
    assert model is not None  # guaranteed by curator_configured() above

    curator_store.start_run(run_id, model, requested, dry_run)

    results: list[AreaResult] = []
    total_cost = 0.0
    budget = CuratorBudget(max_cost)

    for area_name in requested:
        area = KNOWN_AREAS[area_name]
        if budget.is_exhausted():
            results.append(AreaResult(area=area_name, ran=False, skipped_reason="budget exhausted"))
            continue

        # Per-area, cross-process file lock (same filelock-backed primitive
        # mazu.checkpoint.lock already uses, and for the same reason): without
        # this, two concurrent `mazu curator run` invocations on the same area
        # could both pass the eligibility/watermark check before either advances
        # it, both do real (paid) LLM work redundantly. Found live during testing
        # -- never caused data loss (every mutation stays reversible regardless),
        # but wastes a real API call. Held across the ENTIRE eligibility-check
        # through watermark-advance sequence, not just the write, so the check
        # and the act are atomic against another process doing the same area.
        # Different areas use different lock files, so unrelated areas (or a
        # single project's memory vs skills passes) still run concurrently.
        try:
            with ReentrantFileLock(_area_lock_path(root, area_name)).acquire():
                if not is_eligible(curator_store, area, root, forced=forced, full=full):
                    results.append(AreaResult(area=area_name, ran=False, skipped_reason="not eligible (watermark/no signal)"))
                    continue

                memory_store = MemoryStore(_memory_db_path(root)) if area_name in ("memory", "conflicts") else None
                skill_manager = SkillManager(root) if area_name == "skills" else None
                area_action_log_store = (
                    ActionLogStore(_action_log_db_path(root)) if area_name in ("skills", "housekeeping") else None
                )
                area_run_store = RunStore(_runs_db_path(root)) if area_name == "housekeeping" else None
                area_usage_store = UsageStore(_usage_db_path()) if area_name == "housekeeping" else None

                try:
                    registry = build_curator_registry(
                        area=area_name, run_id=run_id, dry_run=dry_run,
                        memory_store=memory_store, curator_store=curator_store,
                        skill_manager=skill_manager, action_log_store=area_action_log_store,
                        root=root, run_store=area_run_store, usage_store=area_usage_store,
                    )
                    pass_result = run_curator_pass(
                        task_prompt=area.task_prompt, system_prompt=CURATOR_SYSTEM_PROMPT,
                        registry=registry, max_rounds=max_rounds, budget=budget, verbose=verbose,
                    )
                except Exception as e:
                    results.append(AreaResult(area=area_name, ran=False, skipped_reason=f"error: {e}"))
                    continue
                finally:
                    if memory_store is not None:
                        memory_store.close()
                    if area_action_log_store is not None:
                        area_action_log_store.close()
                    if area_run_store is not None:
                        area_run_store.close()
                    if area_usage_store is not None:
                        area_usage_store.close()

                provider_name, model_name = _split_model(model)
                area_cost = 0.0
                for usage in pass_result.usage_list:
                    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                    cost = estimate_cost(model, input_tokens, output_tokens)
                    if cost is not None:
                        area_cost += cost
                    usage_store.log("curator", run_id, provider_name, model_name, input_tokens, output_tokens, cost)
                total_cost += area_cost

                area_log_count = len([r for r in curator_store.log_for_run(run_id) if r["area"] == area_name])
                results.append(AreaResult(
                    area=area_name, ran=True, final_text=pass_result.final_text,
                    cost=area_cost, log_entries=area_log_count,
                ))

                # Watermark advances only on a successful, non-dry-run pass -- a
                # crashed or dry pass must re-run next time rather than being
                # silently skipped forever.
                if not dry_run:
                    curator_store.advance_watermark(area_name, run_id)
        except LockTimeoutError:
            results.append(AreaResult(
                area=area_name, ran=False,
                skipped_reason="another curator process is already working on this area",
            ))

    # "completed" only if every requested area either ran cleanly or was
    # legitimately skipped (watermark/signal/budget/lock contention) -- an area
    # that errored (a real API failure, a tool exception) must be visible in the
    # run's own status, not just buried in one AreaResult's skipped_reason, so
    # `mazu curator report`/`mazu curator status` don't quietly say "completed"
    # for a pass where something actually broke.
    any_area_errored = any(
        r.skipped_reason is not None and r.skipped_reason.startswith("error:") for r in results
    )
    final_status = "completed_with_errors" if any_area_errored else "completed"
    curator_store.finish_run(run_id, final_status, None, total_cost)
    curator_store.close()
    usage_store.close()

    return CuratorRunSummary(ran=True, run_id=run_id, areas=results, total_cost=total_cost)
