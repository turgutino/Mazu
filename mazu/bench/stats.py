"""Aggregation for benchmark trials. Deliberately dependency-free."""

import math
from dataclasses import dataclass


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate. Behaves sensibly at the small n a
    handful of repeated trials gives, unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class ConfigSummary:
    config: str
    trials: int
    passed: int
    pass_rate: float
    pass_low: float
    pass_high: float
    mean_cost: float
    cost_stdev: float
    cost_per_solved: float | None
    mean_steps: float
    errored: int
    has_unpriced_calls: bool


def summarize(results: list[dict]) -> list[ConfigSummary]:
    by_config: dict[str, list[dict]] = {}
    for r in results:
        by_config.setdefault(r["config"], []).append(r)

    summaries = []
    for name, rows in by_config.items():
        n = len(rows)
        passed = sum(1 for r in rows if r.get("passed") is True)
        costs = [r.get("cost_usd") or 0.0 for r in rows]
        mean_cost = sum(costs) / n
        stdev = math.sqrt(sum((c - mean_cost) ** 2 for c in costs) / (n - 1)) if n > 1 else 0.0
        low, high = wilson_interval(passed, n)
        steps = [r.get("steps") or 0 for r in rows]
        summaries.append(
            ConfigSummary(
                config=name,
                trials=n,
                passed=passed,
                pass_rate=passed / n,
                pass_low=low,
                pass_high=high,
                mean_cost=mean_cost,
                cost_stdev=stdev,
                cost_per_solved=(sum(costs) / passed) if passed else None,
                mean_steps=sum(steps) / n,
                errored=sum(1 for r in rows if r.get("error")),
                has_unpriced_calls=any(r.get("has_unpriced_calls") for r in rows),
            )
        )
    return summaries


def format_report(summaries: list[ConfigSummary]) -> str:
    header = (
        f"{'config':<22}{'n':>4}{'pass':>6}{'pass rate (95% CI)':>26}"
        f"{'mean $':>10}{'sd $':>9}{'$/solved':>11}{'steps':>7}"
    )
    lines = [header, "-" * len(header)]
    notes = []
    for s in summaries:
        per_solved = f"{s.cost_per_solved:.4f}" if s.cost_per_solved is not None else "n/a"
        rate = f"{s.pass_rate:.0%} ({s.pass_low:.0%}-{s.pass_high:.0%})"
        flag = "*" if s.has_unpriced_calls else ""
        lines.append(
            f"{s.config:<22}{s.trials:>4}{s.passed:>6}{rate:>26}"
            f"{s.mean_cost:>10.4f}{s.cost_stdev:>9.4f}{per_solved:>10}{flag or ' '}{s.mean_steps:>7.1f}"
        )
        if s.errored:
            notes.append(f"{s.config}: {s.errored} trial(s) errored before the agent could finish.")
        if s.has_unpriced_calls:
            notes.append(
                f"{s.config}: * some calls used a model with no pricing entry, so its cost is a lower bound."
            )
    lines.append("")
    lines.append(
        "Intervals are Wilson 95%. With few repeats they are wide -- an apparent gap between two "
        "configs whose intervals overlap is not evidence of a real difference."
    )
    lines.extend(notes)
    return "\n".join(lines)
