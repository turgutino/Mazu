"""Compare two configs in a `mazu bench` results file with a permutation test that shuffles the
config label within each task. Usage:
    python bench/compare_configs.py results.jsonl --a baseline --b protect-tests
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mazu.bench.stats import wilson_interval  # noqa: E402


def load(path: Path, a: str, b: str):
    by_task = defaultdict(lambda: {a: [], b: []})
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["config"] in (a, b):
                by_task[r["task"]][r["config"]].append(bool(r["passed"]))
    return dict(by_task)


def pooled_diff(by_task, a, b) -> float:
    pa = [x for t in by_task.values() for x in t[a]]
    pb = [x for t in by_task.values() for x in t[b]]
    return sum(pb) / len(pb) - sum(pa) / len(pa)


def permutation_p(by_task, a, b, iterations: int, seed: int = 0) -> float:
    rng = random.Random(seed)
    observed = pooled_diff(by_task, a, b)
    hits = 0
    for _ in range(iterations):
        shuffled = {}
        for task, d in by_task.items():
            pool = d[a] + d[b]
            rng.shuffle(pool)
            shuffled[task] = {a: pool[: len(d[a])], b: pool[len(d[a]):]}
        if abs(pooled_diff(shuffled, a, b)) >= abs(observed) - 1e-12:
            hits += 1
    return (hits + 1) / (iterations + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--iterations", type=int, default=20000)
    args = ap.parse_args()
    by_task = load(args.results, args.a, args.b)
    print(f"{'task':<34} {args.a:>14} {args.b:>14}")
    worst_drop = 0.0
    for task, d in sorted(by_task.items()):
        ra = sum(d[args.a]) / max(len(d[args.a]), 1); rb = sum(d[args.b]) / max(len(d[args.b]), 1)
        worst_drop = max(worst_drop, ra - rb)
        print(f"{task:<34} {sum(d[args.a]):>6}/{len(d[args.a]):<7} {sum(d[args.b]):>6}/{len(d[args.b]):<7}")
    for name in (args.a, args.b):
        k = sum(x for t in by_task.values() for x in t[name]); n = sum(len(t[name]) for t in by_task.values())
        lo, hi = wilson_interval(k, n)
        print(f"{name}: {k}/{n} = {k / n:.0%}  (Wilson 95%: {lo:.0%}-{hi:.0%})")
    diff = pooled_diff(by_task, args.a, args.b)
    p = permutation_p(by_task, args.a, args.b, args.iterations)
    print(f"difference ({args.b} - {args.a}): {diff * 100:+.1f} points, two-sided permutation p = {p:.4f} "
          f"(shuffled within tasks, {args.iterations} shuffles)")
    print(f"largest per-task drop for {args.b}: {worst_drop * 100:.0f} points")
    helpful = diff >= 0.10 and p < 0.05 and worst_drop <= 0.20
    print("pre-registered rule:", "HELPFUL" if helpful else "NOT SHOWN TO HELP")


if __name__ == "__main__":
    main()
