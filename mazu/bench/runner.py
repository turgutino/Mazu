"""Runs benchmark trials end to end and records one JSON line per trial.

Each trial gets a fresh `git clone` of the task's repo and its own throwaway HOME, so
(a) trials cannot contaminate each other, (b) nothing a trial does -- memories, usage
rows, config -- touches the real ~/.mazu, and (c) the cost of a trial is read from a
usage database that contains only that trial. The user's API-key config is copied in
so the agent can still authenticate. Pass/fail comes from the task's verify command
run after the agent exits, never from anything the agent reports.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from mazu.bench.spec import Config, Task


def remove_tree(path: Path) -> None:
    """rmtree that also removes read-only files. Git marks object files read-only, and on
    Windows plain shutil.rmtree(..., ignore_errors=True) silently leaves such a clone behind."""
    import stat

    def _onexc(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onexc=_onexc)


def default_run_agent(clone: Path, task: Task, config: Config, env: dict, timeout: int) -> tuple[int, str]:
    cmd = [
        # Drop cwd from sys.path first: when the task repo IS a Mazu checkout (a historical
        # commit), the agent must still run as the Mazu under test, not as that old code.
        sys.executable, "-c",
        "import sys; sys.path[:] = [p for p in sys.path if p not in ('', '.')]; "
        "from mazu.cli import main; main()",
        "run", task.prompt,
        "--max-steps", str(task.max_steps),
        "--allow-shell", "--auto-commit-baseline",
    ]
    if config.model:
        cmd += ["--model", config.model]
    cmd += list(config.extra_args)
    try:
        proc = subprocess.run(
            cmd, cwd=clone, env=env, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return (124, f"agent timed out after {timeout}s")
    return (proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


def _git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _trial_environment(trial_home: Path, config: Config) -> dict:
    env = dict(os.environ)
    real_home = Path.home()
    # git reads user.name/email from HOME; point it at the real file explicitly.
    real_gitconfig = real_home / ".gitconfig"
    if real_gitconfig.exists() and "GIT_CONFIG_GLOBAL" not in env:
        env["GIT_CONFIG_GLOBAL"] = str(real_gitconfig)
    env["HOME"] = str(trial_home)
    env["USERPROFILE"] = str(trial_home)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(config.env)
    return env


def _read_run(clone: Path, trial_home: Path) -> dict:
    """Reads the agent's own bookkeeping for the trial: its run row and its spend."""
    from mazu.runs.store import RunStore
    from mazu.usage.store import UsageStore

    info: dict = {"run_id": None, "run_status": None, "stop_reason": None, "steps": 0,
                  "cost_usd": 0.0, "has_unpriced_calls": False, "calls": 0,
                  "cache_read_tokens": 0, "input_tokens": 0, "output_tokens": 0}
    runs_db = clone / ".mazu" / "runs.db"
    if not runs_db.exists():
        return info
    store = RunStore(runs_db)
    try:
        rows = store.list_runs(1)
        if not rows:
            return info
        row = rows[0]
        info.update(run_id=row["id"], run_status=row["status"],
                    stop_reason=row["stop_reason"], steps=row["last_step"])
    finally:
        store.close()

    usage_db = trial_home / ".mazu" / "usage.db"
    if usage_db.exists() and info["run_id"]:
        usage = UsageStore(usage_db)
        try:
            summary = usage.summary(session_id=info["run_id"])
        finally:
            usage.close()
        info["cost_usd"] = summary["total_cost"]
        info["has_unpriced_calls"] = summary["has_unpriced_calls"]
        info["calls"] = summary["total_calls"]
        info["cache_read_tokens"] = summary["total_cache_read_tokens"]
        info["input_tokens"] = sum(m["input_tokens"] or 0 for m in summary["by_model"])
        info["output_tokens"] = sum(m["output_tokens"] or 0 for m in summary["by_model"])
    return info


def run_trial(
    task: Task,
    config: Config,
    repeat: int,
    work_root: Path,
    run_agent: Callable = default_run_agent,
    agent_timeout: int = 1800,
    verify_timeout: int = 600,
) -> dict:
    started = time.time()
    result: dict = {"task": task.id, "config": config.name, "repeat": repeat,
                    "passed": False, "error": None}
    trial_dir = Path(tempfile.mkdtemp(prefix=f"bench-{task.id}-{config.name}-{repeat}-", dir=work_root))
    clone = trial_dir / "repo"
    trial_home = trial_dir / "home"
    try:
        (trial_home / ".mazu").mkdir(parents=True)
        real_config = Path.home() / ".mazu" / "config.toml"
        if real_config.exists():
            shutil.copy2(real_config, trial_home / ".mazu" / "config.toml")

        _git(["clone", "--quiet", "--local", str(task.repo), str(clone)])
        if task.ref:
            _git(["checkout", "--quiet", task.ref], cwd=clone)

        env = _trial_environment(trial_home, config)
        code, output = run_agent(clone, task, config, env, agent_timeout)
        result["agent_exit_code"] = code
        result["agent_output_tail"] = output[-800:]
        result.update(_read_run(clone, trial_home))

        try:
            verify = subprocess.run(
                task.verify_command, cwd=clone, shell=True, capture_output=True, text=True,
                timeout=verify_timeout, encoding="utf-8", errors="replace", env=env,
            )
            result["passed"] = verify.returncode == 0
            result["verify_output_tail"] = ((verify.stdout or "") + (verify.stderr or ""))[-800:]
        except subprocess.TimeoutExpired:
            result["error"] = f"verify command timed out after {verify_timeout}s"
    except Exception as e:  # a broken trial must be recorded, not abort the whole suite
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["duration_s"] = round(time.time() - started, 1)
        remove_tree(trial_dir)
    return result


def _done_keys(results_path: Path) -> set[tuple]:
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["task"], r["config"], r["repeat"]))
    return done


def run_suite(
    tasks: list[Task],
    configs: list[Config],
    repeats: int,
    results_path: Path,
    run_agent: Callable = default_run_agent,
    agent_timeout: int = 1800,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Runs every (repeat, task, config) combination, appending each finished trial to
    `results_path` immediately, and skipping combinations already recorded there so an
    interrupted suite can be resumed. Configs are interleaved inside each (repeat, task)
    so slow drift in provider latency/behaviour hits every config alike."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(results_path)
    work_root = Path(tempfile.mkdtemp(prefix="mazu-bench-"))
    new_results = []
    try:
        for repeat in range(repeats):
            for task in tasks:
                for config in configs:
                    if (task.id, config.name, repeat) in done:
                        continue
                    result = run_trial(task, config, repeat, work_root, run_agent, agent_timeout)
                    with results_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")
                    new_results.append(result)
                    if on_result:
                        on_result(result)
    finally:
        remove_tree(work_root)
    return new_results


def load_results(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    return [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
