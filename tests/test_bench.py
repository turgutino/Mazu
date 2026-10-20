"""mazu bench: spec loading, statistics and the trial runner, using a fake agent that
writes the same bookkeeping (runs.db row, usage.db rows) a real `mazu run` would."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mazu.bench.runner import load_results, run_suite, run_trial
from mazu.bench.spec import Config, Task, load_configs, load_suite
from mazu.bench.stats import format_report, summarize, wilson_interval
from mazu.runs.store import RunStore
from mazu.usage.store import UsageStore


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "src_repo"
    r.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@t.com"], ["config", "user.name", "T"]):
        subprocess.run(["git", *args], cwd=r, check=True)
    (r / "value.txt").write_text("broken\n")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=r, check=True)
    return r


def _verify_cmd() -> str:
    return f'"{sys.executable}" -c "import sys; sys.exit(0 if open(\'value.txt\').read().strip()==\'fixed\' else 1)"'


def make_fake_agent(fix: bool, cost: float = 0.05, priced: bool = True):
    def fake(clone: Path, task: Task, config: Config, env: dict, timeout: int):

        home = Path(env["USERPROFILE"])
        assert home != Path.home(), "trial must not use the real home directory"
        if fix:
            (clone / "value.txt").write_text("fixed\n")
        (clone / ".mazu").mkdir(exist_ok=True)
        runs = RunStore(clone / ".mazu" / "runs.db")
        runs.start("run-1", task.prompt, config.model, 5, 1, True, None, None, False)
        runs.update_progress("run-1", 3)
        runs.finish("run-1", "completed", "done")
        runs.close()
        usage = UsageStore(home / ".mazu" / "usage.db")
        usage.log("run", "run-1", "anthropic", "m", 1000, 100, cost if priced else None, 400, 0)
        usage.close()
        return (0, "ok")
    return fake


def test_wilson_interval_bounds():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(5, 5)
    assert high == 1.0 and 0.5 < low < 1.0
    low, high = wilson_interval(3, 10)
    assert low < 0.3 < high


def test_load_suite_validates(tmp_path):
    good = tmp_path / "s.json"
    good.write_text(json.dumps([{"id": "a", "repo": "r", "prompt": "p", "verify_command": "v"}]))
    assert load_suite(good)[0].max_steps == 15
    dup = tmp_path / "d.json"
    dup.write_text(json.dumps([{"id": "a", "repo": "r", "prompt": "p", "verify_command": "v"}] * 2))
    with pytest.raises(ValueError, match="Duplicate"):
        load_suite(dup)
    missing = tmp_path / "m.json"
    missing.write_text(json.dumps([{"id": "a", "repo": "r", "prompt": "p"}]))
    with pytest.raises(ValueError, match="verify_command"):
        load_suite(missing)


def test_load_configs(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"name": "base"}, {"name": "x", "model": "m", "extra_args": ["--a"], "env": {"K": 1}}]))
    configs = load_configs(p)
    assert configs[1].extra_args == ("--a",) and configs[1].env == {"K": "1"}


def test_trial_passes_when_verify_passes_and_reads_cost(repo, tmp_path):
    task = Task("t", str(repo), "fix it", _verify_cmd())
    result = run_trial(task, Config("base"), 0, tmp_path, run_agent=make_fake_agent(True, 0.05))
    assert result["passed"] is True and result["error"] is None
    assert result["cost_usd"] == pytest.approx(0.05)
    assert result["steps"] == 3 and result["run_status"] == "completed"
    assert result["cache_read_tokens"] == 400 and result["has_unpriced_calls"] is False


def test_agent_claiming_success_does_not_count_if_verify_fails(repo, tmp_path):
    task = Task("t", str(repo), "fix it", _verify_cmd())
    result = run_trial(task, Config("base"), 0, tmp_path, run_agent=make_fake_agent(False))
    assert result["passed"] is False and result["run_status"] == "completed"


def test_source_repo_is_never_modified(repo, tmp_path):
    task = Task("t", str(repo), "fix it", _verify_cmd())
    run_trial(task, Config("base"), 0, tmp_path, run_agent=make_fake_agent(True))
    assert (repo / "value.txt").read_text() == "broken\n"


def test_unpriced_cost_is_flagged(repo, tmp_path):
    task = Task("t", str(repo), "fix it", _verify_cmd())
    result = run_trial(task, Config("base"), 0, tmp_path, run_agent=make_fake_agent(True, priced=False))
    assert result["has_unpriced_calls"] is True


def test_agent_crash_is_recorded_not_raised(repo, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("provider exploded")
    task = Task("t", str(repo), "fix it", _verify_cmd())
    result = run_trial(task, Config("base"), 0, tmp_path, run_agent=boom)
    assert result["passed"] is False and "provider exploded" in result["error"]


def test_suite_interleaves_appends_and_resumes(repo, tmp_path):
    tasks = [Task("t1", str(repo), "p", _verify_cmd()), Task("t2", str(repo), "p", _verify_cmd())]
    configs = [Config("a"), Config("b")]
    out = tmp_path / "res" / "r.jsonl"
    seen = []
    run_suite(tasks, configs, 2, out, run_agent=make_fake_agent(True), on_result=lambda r: seen.append(r))
    assert len(seen) == 8
    assert [(r["repeat"], r["task"], r["config"]) for r in seen[:4]] == [
        (0, "t1", "a"), (0, "t1", "b"), (0, "t2", "a"), (0, "t2", "b")]
    again = run_suite(tasks, configs, 2, out, run_agent=make_fake_agent(True))
    assert again == [] and len(load_results(out)) == 8
    more = run_suite(tasks, configs, 3, out, run_agent=make_fake_agent(True))
    assert len(more) == 4


def test_summary_and_report():
    rows = (
        [{"config": "a", "passed": True, "cost_usd": 0.1, "steps": 4}] * 3
        + [{"config": "a", "passed": False, "cost_usd": 0.3, "steps": 9, "has_unpriced_calls": True}]
        + [{"config": "b", "passed": False, "cost_usd": 0.2, "steps": 5, "error": "x"}]
    )
    by = {s.config: s for s in summarize(rows)}
    assert by["a"].passed == 3 and by["a"].pass_rate == 0.75
    assert by["a"].cost_per_solved == pytest.approx(0.6 / 3)
    assert by["b"].cost_per_solved is None and by["b"].errored == 1
    text = format_report(list(by.values()))
    assert "lower bound" in text and "n/a" in text and "Wilson" in text


def test_cli_report_and_validation(tmp_path):
    from click.testing import CliRunner

    from mazu.cli import main

    res = tmp_path / "r.jsonl"
    res.write_text(json.dumps({"task": "t", "config": "a", "repeat": 0, "passed": True, "cost_usd": 0.1, "steps": 3}) + "\n")
    out = CliRunner().invoke(main, ["bench", "report", str(res)])
    assert out.exit_code == 0 and "$/solved" in out.output and "a" in out.output

    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    out = CliRunner().invoke(main, ["bench", "run", "--suite", str(bad), "--configs", str(bad)])
    assert out.exit_code != 0 and "non-empty" in out.output
