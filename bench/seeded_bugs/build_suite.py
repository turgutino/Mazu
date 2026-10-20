"""Builds the seeded-bugs benchmark: one git repo per task plus suite.json.

    python bench/seeded_bugs/build_suite.py --out ./bench-work/seeded-bugs
    python bench/seeded_bugs/build_suite.py --out ./bench-work/seeded-bugs --check

The hidden tests are written next to the repos (not inside them) and run by each task's
verify_command with the trial's clone as the working directory, so the agent cannot see
or edit them. `--check` proves every task is a fair one for free (no model calls): the
untouched repo must FAIL its hidden test and the reference fix must PASS it.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS  # noqa: E402

HIDDEN_HEADER = "import os, sys\nsys.path.insert(0, os.getcwd())\n"
HIDDEN_FOOTER = "\n\nif __name__ == '__main__':\n    import unittest\n    unittest.main()\n"


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def build(out: Path) -> list[dict]:
    if out.exists():
        shutil.rmtree(out)
    (out / "repos").mkdir(parents=True)
    (out / "hidden").mkdir()
    suite = []
    for t in TASKS.values():
        repo = out / "repos" / t["id"]
        repo.mkdir()
        for name, content in t["files"].items():
            (repo / name).write_text(content, encoding="utf-8", newline="\n")
        _git(["init", "-q"], repo)
        _git(["config", "user.email", "bench@example.invalid"], repo)
        _git(["config", "user.name", "bench"], repo)
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "initial"], repo)

        hidden = out / "hidden" / f"{t['id']}.py"
        hidden.write_text(HIDDEN_HEADER + t["hidden_test"] + HIDDEN_FOOTER, encoding="utf-8", newline="\n")
        suite.append({
            "id": t["id"],
            "repo": str(repo.resolve()),
            "prompt": t["prompt"],
            "verify_command": f'"{sys.executable}" "{hidden.resolve()}"',
            "max_steps": 12,
        })
    (out / "suite.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return suite


def check(out: Path, suite: list[dict]) -> bool:
    ok = True
    for item in suite:
        t = TASKS[item["id"]]
        scratch = out / "check" / t["id"]
        shutil.copytree(item["repo"], scratch)

        def verify():
            return subprocess.run(item["verify_command"], cwd=scratch, shell=True,
                                  capture_output=True, text=True).returncode == 0

        broken_fails = not verify()
        for name, content in t["fixed"].items():
            (scratch / name).write_text(content, encoding="utf-8", newline="\n")
        fixed_passes = verify()
        status = "ok" if (broken_fails and fixed_passes) else "BAD"
        print(f"{status:>4}  {t['id']:<26} broken-fails={broken_fails} fixed-passes={fixed_passes}")
        ok = ok and broken_fails and fixed_passes
    shutil.rmtree(out / "check", ignore_errors=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", action="store_true", help="verify tasks are fair; makes no model calls")
    args = ap.parse_args()
    suite = build(args.out)
    print(f"built {len(suite)} tasks in {args.out}")
    if args.check:
        return 0 if check(args.out, suite) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
