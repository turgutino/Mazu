"""Builds a benchmark from Mazu's own git history: real bugs, real regression tests.

Each task checks the repo out at the PARENT of a historical bug-fix commit, gives the
agent a symptom-only description, and judges the result with the regression test file(s)
that the real fix added. Those test files are kept OUTSIDE the clone (hidden/) and copied
over the clone right before they run, so the agent can neither read nor edit them.

    python bench/mazu_history/build_suite.py --out ./bench-work/mazu-history
    python bench/mazu_history/build_suite.py --out ./bench-work/mazu-history --check

`--check` makes no model calls: at the parent commit every hidden test must FAIL, at the
fix commit it must PASS. A prompt is only allowed to name an interface (a function, a
message) when the hidden test requires that exact name -- otherwise the task would be
unfair, because the agent could not possibly guess it.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mazu.bench.runner import remove_tree  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

TASKS = [
    {
        "id": "restore-open-connection",
        "commit": "ab3c5d1",
        "tests": ["tests/test_checkpoint.py"],
        "prompt": (
            "Bug: CheckpointManager.restore() can report success while memory.db is NOT actually "
            "reverted. Reproduction: a project with no .gitignore entry for .mazu/ (e.g. the "
            "CheckpointManager is built directly by an embedding app). Add a memory, take a "
            "checkpoint, add a second memory, keep an idle MemoryStore connection to memory.db open, "
            "then call restore(). Afterwards a fresh connection should show only the first memory, "
            "but the restore silently does nothing for memory.db. Restoring must either truly revert "
            "memory.db or fail loudly, never silently succeed. Find the root cause in "
            "mazu/checkpoint/ and fix it."
        ),
    },
    {
        "id": "doctor-config-key",
        "commit": "50784fa",
        "tests": ["tests/test_diagnostics.py"],
        "prompt": (
            "Bug: after saving a key with `mazu config set deepseek_api_key ...` (no API-key "
            "environment variable set), `mazu doctor` still FAILs with 'no provider has a key set', "
            "even though `mazu chat` works fine with that same saved key. The doctor's API key check "
            "(check_api_keys in mazu/diagnostics.py) should agree with how every other command "
            "resolves keys. Fix it."
        ),
    },
    {
        "id": "windows-non-ascii-output",
        "commit": "2a3d88d",
        "tests": ["tests/test_shell_safety.py", "tests/test_skills_manager.py"],
        "prompt": (
            "Bug (Windows): when a Python subprocess started by the run_shell tool "
            "(mazu/tools/shell.py) or by SkillManager.run (mazu/skills/manager.py) prints non-ASCII "
            "text such as an emoji or the Azerbaijani letter 'ı', the subprocess itself crashes with "
            "UnicodeEncodeError, because its stdout falls back to the console's legacy codepage. Both "
            "should return the printed text intact (UTF-8; undecodable bytes replaced, never a crash). "
            "Fix both places."
        ),
    },
    {
        "id": "timeline-no-change-message",
        "commit": "b8bc333",
        "tests": ["tests/test_cli.py"],
        "prompt": (
            "Bug: `mazu timeline` prints '(first checkpoint — nothing to compare against)' for ANY "
            "checkpoint whose files_changed list is empty, even when it has a real predecessor in the "
            "same session and that round simply touched no tracked file (a read-only tool round). Those "
            "are different facts. A checkpoint with a real predecessor and no file changes must print "
            "the text 'no files changed since the previous checkpoint' and must not mention 'first "
            "checkpoint'; the genuine first checkpoint keeps its existing message unchanged. Entries "
            "written before branching existed may lack parent_checkpoint_id: treat index 0 as the root "
            "for those. Fix `mazu timeline`."
        ),
    },
    {
        "id": "server-reason-relative-root",
        "commit": "1507b8f",
        "tests": ["tests/test_shell_safety.py"],
        "prompt": (
            "Bug: long_running_server_reason(command, root) in mazu/tools/shell.py correctly flags "
            "`python app.py` when app.py contains a blocking server call and root is absolute, but "
            "returns None when root is a relative path such as Path('.') (with the working directory "
            "set to the project). It must behave the same for relative and absolute roots. Fix it "
            "without changing its signature."
        ),
    },
]

VERIFY = '''"""Copies the hidden test files over the clone, then runs them. Exit 0 = pass."""
import shutil
import subprocess
import sys
from pathlib import Path

hidden_dir, *tests = sys.argv[1:]
cwd = Path.cwd()
for t in tests:
    dst = cwd / t
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(hidden_dir) / t, dst)
code = (
    "import sys; sys.path.insert(0, ''); import mazu, pathlib; "
    "assert pathlib.Path(mazu.__file__).resolve().is_relative_to(pathlib.Path.cwd().resolve()), "
    "'tests are not running against the clone: ' + mazu.__file__; "
    "import pytest; sys.exit(pytest.main(['-q', '-x', '-p', 'no:cacheprovider'] + sys.argv[1:]))"
)
sys.exit(subprocess.run([sys.executable, "-c", code, *tests]).returncode)
'''


def _git(args, cwd=None, **kw):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, **kw)


def build(out: Path) -> list[dict]:
    if out.exists():
        remove_tree(out)
    (out / "hidden").mkdir(parents=True)
    verify_py = out / "verify_history.py"
    verify_py.write_text(VERIFY, encoding="utf-8")
    suite = []
    for t in TASKS:
        parent = _git(["rev-parse", f"{t['commit']}^"], cwd=REPO).stdout.strip()
        full = _git(["rev-parse", t["commit"]], cwd=REPO).stdout.strip()
        for path in t["tests"]:
            dst = out / "hidden" / t["id"] / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            blob = subprocess.run(["git", "show", f"{full}:{path}"], cwd=REPO, check=True, capture_output=True).stdout
            dst.write_bytes(blob)
        suite.append({
            "id": t["id"],
            "repo": str(REPO),
            "ref": parent,
            "prompt": t["prompt"],
            "verify_command": f'"{sys.executable}" "{verify_py}" "{out / "hidden" / t["id"]}" ' + " ".join(t["tests"]),
            "max_steps": 25,
            "_fix_commit": full,
        })
    clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in suite]
    (out / "suite.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    (out / "fix_commits.json").write_text(json.dumps({s["id"]: s["_fix_commit"] for s in suite}, indent=2))
    return suite


def check(out: Path, suite: list[dict]) -> bool:
    ok = True
    for s in suite:
        scratch = out / "check" / s["id"]
        scratch.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--quiet", "--local", s["repo"], str(scratch)])
        _git(["checkout", "--quiet", s["ref"]], cwd=scratch)

        def verify() -> bool:
            return subprocess.run(s["verify_command"], cwd=scratch, shell=True, capture_output=True).returncode == 0

        broken_fails = not verify()
        # The reference fix is the real commit's NON-test changes; tests come from hidden/.
        _git(["checkout", "--quiet", "--force", s["_fix_commit"]], cwd=scratch)
        fixed_passes = verify()
        status = "ok" if (broken_fails and fixed_passes) else "BAD"
        print(f"{status:>4}  {s['id']:<30} parent-fails={broken_fails} fix-passes={fixed_passes}")
        ok = ok and broken_fails and fixed_passes
    remove_tree(out / "check")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", action="store_true", help="verify tasks are fair; makes no model calls")
    args = ap.parse_args()
    out = args.out.resolve()
    suite = build(out)
    print(f"built {len(suite)} tasks in {out}")
    return (0 if check(out, suite) else 1) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
