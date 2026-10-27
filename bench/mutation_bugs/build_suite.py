"""Builds a benchmark of realistic bugs injected into Mazu's own code, judged by hidden tests.

    python bench/mutation_bugs/build_suite.py --out ./bench-work/mutation-bugs
    python bench/mutation_bugs/build_suite.py --out ./bench-work/mutation-bugs --check

Each task is a fresh single-commit repository (so the agent cannot read the injection out of
git history) holding the current Mazu source with ONE bug injected by an exact text
replacement, and without the test file(s) that would expose it. The hidden tests live outside
the repository and are copied over it only when the result is judged, so the agent can neither
see nor edit them. The prompt describes only the symptom.

`--check` makes no model calls and proves each task is fair: with the bug present the hidden
tests must FAIL, and with the original source restored they must PASS.

Honest limits: these bugs are synthetic (a person chose them), they sit in a codebase the
model may have seen, and one-line mutations are easier to localize than most real bugs.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mazu.bench.runner import remove_tree  # noqa: E402
from mazu_history.build_suite import VERIFY  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# `hidden`: {path: None} copies that existing test file from HEAD (and removes it from the task
# repo); {path: "<source>"} is a new test written for this task.
TASKS = [
    {
        "id": "bm25-negative-scores",
        "file": "mazu/memory/bm25.py",
        "old": "math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)",
        "new": "math.log((self.n_docs - freq + 0.5) / (freq + 0.5))",
        "hidden": {"tests/test_hidden_bm25_negative.py": """
from mazu.memory.bm25 import BM25


def test_scores_are_never_negative_when_a_term_is_in_most_documents():
    docs = ["alpha beta", "alpha gamma", "alpha delta", "alpha epsilon", "zeta"]
    assert all(s >= 0 for s in BM25(docs).score("alpha"))


def test_matching_more_query_words_still_ranks_higher():
    docs = ["alpha beta", "alpha", "gamma", "alpha gamma"]
    scores = BM25(docs).score("alpha beta")
    assert scores[0] > scores[1] > 0
"""},
        "prompt": (
            "Bug in memory ranking: when a query contains a word that appears in most of the stored "
            "memories, some memories get a NEGATIVE relevance score, so a memory that matches the query "
            "can rank below one that matches nothing. Relevance scores must never be negative. Find the "
            "cause in the memory ranking code and fix it."
        ),
    },
    {
        "id": "bm25-length-normalisation",
        "file": "mazu/memory/bm25.py",
        "old": "self.b * doc_len / (self.avg_len or 1)",
        "new": "self.b * (self.avg_len or 1) / (doc_len or 1)",
        "hidden": {"tests/test_hidden_bm25_length.py": """
from mazu.memory.bm25 import BM25


def test_shorter_document_outranks_a_longer_one_with_the_same_match():
    docs = ["alpha", "alpha " + "filler " * 40, "unrelated words here"]
    scores = BM25(docs).score("alpha")
    assert scores[0] > scores[1] > 0
"""},
        "prompt": (
            "Bug in memory ranking: a short memory and a very long memory that each contain the query word "
            "exactly once should not score the way they do now -- the long, padded memory currently "
            "outranks (or ties with) the short, focused one, when the short one should clearly win. Fix the "
            "ranking so document length is accounted for in the right direction."
        ),
    },
    {
        "id": "staleness-young-memory-flagged",
        "file": "mazu/memory/staleness.py",
        "old": 'if row["created_at"] >= cutoff:',
        "new": 'if row["created_at"] <= cutoff:',
        "hidden": {"tests/test_memory_staleness.py": None},
        "prompt": (
            "Bug: `mazu memory stale` flags memories that were created only a moment ago as stale, while "
            "memories that have sat unused for months are never listed. A memory younger than the staleness "
            "threshold must never be reported as stale. Find and fix the cause."
        ),
    },
    {
        "id": "staleness-recent-mistakes-archived",
        "file": "mazu/memory/staleness.py",
        "old": '        if row["id"] in floor_mistake_ids:\n            continue\n',
        "new": "",
        "hidden": {"tests/test_memory_staleness.py": None},
        "prompt": (
            "Bug: `mazu memory stale --auto` archives the three most recent 'mistake' memories even though "
            "Mazu injects those into every session on purpose. They must be excluded from staleness the "
            "same way pinned memories are, while older mistakes stay eligible. Find and fix the cause."
        ),
    },
    {
        "id": "consolidate-keeps-oldest",
        "file": "mazu/memory/consolidate.py",
        "old": "        survivor = cluster_sorted[-1]\n        losers = cluster_sorted[:-1]\n",
        "new": "        survivor = cluster_sorted[0]\n        losers = cluster_sorted[1:]\n",
        "hidden": {"tests/test_memory_consolidate.py": None},
        "prompt": (
            "Bug: `mazu memory consolidate` keeps the OLDEST memory of each duplicate group and retires the "
            "newest ones. It is documented to keep the most recently created memory and mark the others as "
            "superseded by it. Fix it."
        ),
    },
    {
        "id": "allowlist-matches-name-prefix",
        "file": "mazu/tools/shell.py",
        "old": 'rf"{re.escape(name)}\\b"',
        "new": 'rf"{re.escape(name)}"',
        "hidden": {"tests/test_hidden_allowlist_prefix.py": """
from mazu.tools.shell import is_allowed_by_shell_allowlist as allowed


def test_only_the_exact_program_name_is_allowed():
    assert allowed("git status", ["git"])
    assert allowed("git", ["git"])
    assert not allowed("gitx status", ["git"])
    assert not allowed("github-cli auth", ["git"])
    assert not allowed("echo git", ["git"])
"""},
        "prompt": (
            "Bug: with a shell allowlist of just `git`, a command such as `gitx status` or `github-cli auth` "
            "(a different program whose name merely STARTS with 'git') is allowed to run. Only commands whose "
            "first word is exactly an allowed program may pass. Find and fix the cause."
        ),
    },
    {
        "id": "pricing-cache-read-uses-output-rate",
        "file": "mazu/llm/pricing.py",
        "old": "(cache_read_tokens / 1_000_000) * input_rate * CACHE_READ_RATE_MULTIPLIER",
        "new": "(cache_read_tokens / 1_000_000) * output_rate * CACHE_READ_RATE_MULTIPLIER",
        "hidden": {"tests/test_hidden_pricing_cache.py": """
import pytest

from mazu.llm.pricing import CACHE_READ_RATE_MULTIPLIER, PRICING_PER_MILLION_TOKENS, estimate_cost


def test_cache_reads_are_priced_from_the_input_rate():
    model = "anthropic:claude-sonnet-5"
    input_rate, output_rate = PRICING_PER_MILLION_TOKENS[model]
    assert input_rate != output_rate
    cost = estimate_cost(model, 0, 0, cache_read_tokens=1_000_000)
    assert cost == pytest.approx(input_rate * CACHE_READ_RATE_MULTIPLIER)
"""},
        "prompt": (
            "Bug: the estimated cost of calls that read from the prompt cache is too high for models whose "
            "output price is much higher than their input price. Cached input tokens are supposed to cost a "
            "small fraction of the model's normal INPUT rate. Find the cause in the cost estimate and fix it."
        ),
    },
    {
        "id": "fs-sibling-folder-escape",
        "file": "mazu/tools/fs.py",
        "old": "    if resolved != root and root not in resolved.parents:\n        raise ValueError",
        "new": "    if resolved != root and not str(resolved).startswith(str(root)):\n        raise ValueError",
        "hidden": {"tests/test_hidden_fs_sibling.py": """
import pytest

from mazu.tools.fs import _safe_path


def test_sibling_folder_sharing_the_root_name_prefix_is_rejected(tmp_path):
    root = (tmp_path / "proj").resolve()
    root.mkdir()
    (tmp_path / "proj-secrets").mkdir()
    with pytest.raises(ValueError):
        _safe_path(root, "../proj-secrets/key.txt")
    with pytest.raises(ValueError):
        _safe_path(root, "../outside.txt")
    assert _safe_path(root, "a/b.txt") == (root / "a" / "b.txt").resolve()
    assert _safe_path(root, ".") == root
"""},
        "prompt": (
            "Bug in the file tools' sandbox: a path such as `../myproject-secrets/key.txt` (a SIBLING folder "
            "whose name starts with the project folder's name) is accepted as being inside the project. "
            "Paths must be rejected unless they really are inside the project root. Find the check in "
            "`mazu/tools/fs.py` and fix it."
        ),
    },
    {
        "id": "dedup-title-case-sensitive",
        "file": "mazu/memory/store.py",
        "old": "normalized = title.strip().lower()",
        "new": "normalized = title.strip()",
        "hidden": {"tests/test_hidden_dedup_case.py": """
from mazu.memory.store import MemoryStore


def test_title_match_ignores_case_and_surrounding_whitespace(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add(category="decision", title="Use PostgreSQL", body="The project uses PostgreSQL.",
              source="explicit", session_id="s")
    assert store.find_duplicate("decision", "Use PostgreSQL") is not None
    assert store.find_duplicate("decision", "  use postgresql ") is not None
    assert store.find_duplicate("decision", "Use SQLite") is None
    assert store.find_duplicate("fact", "Use PostgreSQL") is None
    store.close()
"""},
        "prompt": (
            "Bug: when a memory with capital letters in its title (for example 'Use PostgreSQL') is "
            "extracted again in a later session, Mazu does not recognise it as a duplicate of the one "
            "already stored, so near-identical memories pile up. Title matching is supposed to ignore case "
            "and surrounding spaces. Find the cause in the memory store and fix it."
        ),
    },
]


def _run(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def _git(args, cwd):
    r = _run(["git", *args], cwd)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr}")
    return r


def build(out: Path) -> list[dict]:
    if out.exists():
        remove_tree(out)
    (out / "repos").mkdir(parents=True)
    verify_py = out / "verify_mutation.py"
    verify_py.write_text(VERIFY, encoding="utf-8")
    head = _git(["rev-parse", "HEAD"], REPO).stdout.strip()
    suite = []
    for t in TASKS:
        repo = out / "repos" / t["id"]
        repo.mkdir()
        archive = out / f"{t['id']}.tar"
        _git(["archive", "--format=tar", f"--output={archive}", head], REPO)
        subprocess.run(["tar", "-xf", str(archive), "-C", str(repo)], check=True)
        archive.unlink()

        target = repo / t["file"]
        original = target.read_text(encoding="utf-8")
        if original.count(t["old"]) != 1:
            raise SystemExit(f"{t['id']}: expected exactly one match for the replaced text in {t['file']}, "
                             f"found {original.count(t['old'])}")
        target.write_text(original.replace(t["old"], t["new"]), encoding="utf-8", newline="")

        hidden_dir = out / "hidden" / t["id"]
        for rel, source in t["hidden"].items():
            dst = hidden_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if source is None:  # an existing test: hide it from the task repo
                shutil.copyfile(repo / rel, dst)
                (repo / rel).unlink()
            else:
                dst.write_text(source.lstrip("\n"), encoding="utf-8")

        for args in (["init", "-q"], ["config", "user.email", "bench@example.invalid"],
                     ["config", "user.name", "bench"], ["add", "-A"], ["commit", "-q", "-m", "initial"]):
            _git(args, repo)
        suite.append({
            "id": t["id"],
            "repo": str(repo.resolve()),
            "prompt": t["prompt"],
            "verify_command": f'"{sys.executable}" "{verify_py.resolve()}" "{hidden_dir.resolve()}" '
                              + " ".join(t["hidden"]),
            "max_steps": 25,
            "_original": original,
            "_file": t["file"],
        })
    clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in suite]
    (out / "suite.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return suite


def check(out: Path, suite: list[dict]) -> bool:
    ok = True
    for s in suite:
        scratch = out / "check" / s["id"]
        scratch.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(s["repo"], scratch)

        def verify():
            r = subprocess.run(s["verify_command"], cwd=scratch, shell=True, capture_output=True, text=True)
            return r.returncode == 0, r.stdout[-300:]

        with_bug, tail = verify()
        (scratch / s["_file"]).write_text(s["_original"], encoding="utf-8", newline="")
        restored, _ = verify()
        status = "ok" if (not with_bug and restored) else "BAD"
        print(f"{status:>4}  {s['id']:<40} bug-fails={not with_bug} original-passes={restored}")
        if status == "BAD":
            print("      last output with the bug present:", tail.replace("\n", " | ")[-250:])
        ok = ok and not with_bug and restored
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
