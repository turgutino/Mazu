# Benchmark results

Measured on 2026-10-27 and 2026-10-28 with `mazu bench`, model `deepseek:deepseek-chat`, plain `mazu run` unless a config says otherwise, `--max-cost 0.60` per trial, shell commands allowed inside a throwaway clone. Every trial runs in a fresh `git clone` with its own home directory and is judged only by hidden tests copied in after the agent finishes. Intervals are 95% Wilson intervals.

These are small suites. Read [What these numbers do not show](#what-these-numbers-do-not-show) before quoting them.

## Summary

| Suite | Trials per config | `baseline` | `protect-tests` |
|---|---|---|---|
| `mazu_history`: 5 real bugs from this repository's history | 35 | **25 / 35 = 71%** (55%-84%) | 24 / 35 = 69% (52%-81%) |
| `mutation_bugs`: 9 one-line bugs injected into current code | 27 | **27 / 27 = 100%** (88%-100%) | 24 / 27 = 89% (72%-96%) |

`protect-tests` (`mazu run --protect-tests`, a system-prompt rule telling the agent not to edit existing tests) is **not shown to help**: under the decision rule fixed in advance in [analysis_plan.md](analysis_plan.md) it needed at least +10 points, p < 0.05, and no task dropping by more than 20 points. It got -2.9 points (p = 1.00) on the real bugs and -11.1 points (p = 0.20) on the injected bugs, so the rule was not met on either suite.

## Real bugs (`mazu_history`)

Each task checks this repository out at the commit before a real bug fix, describes only the symptom, and is judged by the regression test the real fix added (kept hidden). Confirmation run, 7 repeats per task per config:

| Task | `baseline` | `protect-tests` |
|---|---|---|
| doctor-config-key | 6 / 7 | 7 / 7 |
| restore-open-connection | **0 / 7** | **0 / 7** |
| server-reason-relative-root | 6 / 7 | 7 / 7 |
| timeline-no-change-message | 7 / 7 | 6 / 7 |
| windows-non-ascii-output | 6 / 7 | 4 / 7 |

- Cost: `baseline` $0.037 per trial and $0.051 per solved task; `protect-tests` $0.043 and $0.063. Median trial time 171 s and 228 s.
- 27 of 35 trials in each config ended at the 25-step limit rather than by the agent stopping, and 18 of those were passing trials (of 25 and 24 passes). The agent often does not stop on its own when the work is done; the trials do not show whether it had already fixed the bug earlier.
- `restore-open-connection` (an open SQLite connection makes `rollback` silently not revert the memory database) was never solved: 0 of 22 trials across the pilot and the confirmation run, in both configs.
- **Pilot, exploratory only.** An earlier, smaller run had `baseline` at 10 / 15 and `protect-tests` at 19 / 25. That gap came mostly from one task (`doctor-config-key`, 1 / 3 against 4 / 5) and did not reproduce in the confirmation run, which is the reason the confirmation run was pre-registered. The pilot trials are not included in the table above.

## Injected bugs (`mutation_bugs`)

Nine one-line bugs injected into Mazu's own code (inverted comparison, wrong rate, dropped `.lower()`, and similar), each in a single-commit repository so the injection cannot be read from git history, with the test that exposes it removed and kept hidden. 3 repeats per task per config:

| Task | `baseline` | `protect-tests` |
|---|---|---|
| allowlist-matches-name-prefix | 3 / 3 | 3 / 3 |
| bm25-length-normalisation | 3 / 3 | 3 / 3 |
| bm25-negative-scores | 3 / 3 | 1 / 3 |
| consolidate-keeps-oldest | 3 / 3 | 3 / 3 |
| dedup-title-case-sensitive | 3 / 3 | 3 / 3 |
| fs-sibling-folder-escape | 3 / 3 | 3 / 3 |
| pricing-cache-read-uses-output-rate | 3 / 3 | 3 / 3 |
| staleness-recent-mistakes-archived | 3 / 3 | 2 / 3 |
| staleness-young-memory-flagged | 3 / 3 | 3 / 3 |

`baseline` solved everything, so this suite is at its ceiling and cannot separate configurations: it tells us the agent reliably finds and fixes a localized, well-described bug, not how it compares with anything. Cost $0.022 per trial for `baseline`; 6 of 27 trials (all passing) ended at the step limit, against 11 of 27 for `protect-tests`.

## System performance (local, no model calls)

`python bench/perf_local.py` measures the parts of Mazu that do not depend on a model, in a throwaway directory. Timings are from one Windows machine and varied by up to 2x between two runs of the same script (e.g. 0.5 s versus 1.0 s for the 10,000-memory case), so read them as orders of magnitude. The memory text is synthetic.

| Memories stored | Build the context block (every session start) | Duplicate check (per extracted memory) | Database size |
|---|---|---|---|
| 100 | 7-9 ms | about 1 ms | 0.1 MB |
| 1,000 | 50-91 ms | 13-15 ms | 0.6 MB |
| 5,000 | 250-500 ms | 63-135 ms | 2.9 MB |
| 10,000 | 0.5-1.0 s | 136-284 ms | 5.7 MB |

Retrieval ranks every active memory with BM25 on each session start, so the cost grows linearly with the number of memories (there is no index). The text injected into the prompt stays bounded (about 6,900 characters here) regardless.

| Project size | Checkpoint (snapshot) | Rollback | Disk per checkpoint |
|---|---|---|---|
| 50 files | 260-380 ms | 150-250 ms | 0.17 MB |
| 500 files | 250-350 ms | 160-190 ms | 0.17 MB |
| 3,000 files | 270-420 ms | 260-320 ms | 0.17 MB |

Snapshot and rollback time is dominated by launching git, not by project size, up to the sizes tested. Disk per checkpoint is mostly the copy of the memory database (here 300 memories); with the default retention of 50 checkpoints, a 10,000-memory database (5.7 MB) would use roughly 290 MB. That last figure is arithmetic, not a measurement.

**Interrupted rollback.** `restore()` has 7 steps and is not atomic. Making each step fail in turn (a simulation of a failure between steps, not of a process being killed mid-write), 4 of the 7 leave code, memory, and skills out of step with each other, and in all 4 cases running the rollback again produced a fully consistent state. This is now a test: `tests/test_checkpoint_restore_interruption.py`.

`mazu --version` takes 0.8-1.1 s to start (median over 5 runs per script run); the cause was not investigated.

## What these numbers do not show

- **Small and narrow.** Five real bugs and nine injected ones, all in one repository, one model, one language.
- **Possible contamination.** The repository is public, so the model may have seen it. Injected bugs are synthetic and were chosen by the person who built the suite; real bugs are harder to localize than one-line mutations.
- **No comparison with anything else.** Not another coding agent, not a stronger model on the same tasks. "71%" is a measured baseline for judging Mazu changes, not a ranking.
- **Task clustering.** Trials of one task are not independent samples of "coding tasks"; the permutation test shuffles within tasks to account for that, but the intervals above treat trials as independent and are therefore optimistic about generalization.
- **One run each.** Provider behavior drifts over days; both configs were interleaved within the same run to share that drift, but the results are from one session.

## Reproduce

```bash
python bench/mazu_history/build_suite.py --out ./bench-work/mazu-history --check
python bench/mutation_bugs/build_suite.py --out ./bench-work/mutation-bugs --check
# configs.json: [{"name": "baseline", "model": "deepseek:deepseek-chat", "extra_args": ["--max-cost", "0.60"]},
#                {"name": "protect-tests", "model": "deepseek:deepseek-chat", "extra_args": ["--max-cost", "0.60", "--protect-tests"]}]
mazu bench run --suite bench-work/mazu-history/suite.json --configs configs.json --repeats 7 --workers 4 --out hist.jsonl
python bench/compare_configs.py hist.jsonl --a baseline --b protect-tests
```

`mazu bench run` makes real, billed model calls (about $1.3 to $1.5 per config for the confirmation run above).
