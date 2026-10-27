# Analysis plan for the confirmation run (written before the run starts)

Date written: 2026-10-27, before any confirmation trial exists.

## What came before (pilot, exploratory only)

Two earlier runs on the same five-task suite (`bench/mazu_history`) exist: `baseline` 10/15 and `protect-tests` 19/25. They were looked at before this plan was written, so they are **pilot data** and are not part of any test below. The `--protect-tests` prompt rule was chosen *because* of what the baseline showed, which is a reason to distrust the pilot gap.

## The confirmation run

- Suite: `bench/mazu_history` (5 real bugs from this repository's history, hidden regression tests).
- Model: `deepseek:deepseek-chat`, both configs, `--max-cost 0.60` per trial.
- Configs: `baseline` (plain `mazu run`) and `protect-tests` (`mazu run --protect-tests`).
- 7 repeats per task per config = 35 trials per config, run in a fresh results file, workers in parallel. Trial order is repeat-major and configs are interleaved.

## Primary question

Does `--protect-tests` change the pass rate?

- Statistic: pooled pass rate of `protect-tests` minus pooled pass rate of `baseline` (the design is balanced, so this equals the mean of the per-task differences).
- Test: permutation test that shuffles the config label **within each task** (this respects that tasks differ a lot in difficulty), 20,000 shuffles, two-sided p-value. Script: `bench/compare_configs.py`.
- Decision rule, fixed now: the flag is described as **helpful** only if the pooled difference is at least +10 points, the two-sided p-value is below 0.05, and no task's pass rate drops by more than 20 points. Otherwise it is reported as **not shown to help**, whatever the direction. It is not made the default in either case without a separate decision.

## Reported regardless of outcome

Pass rate with Wilson 95% interval per config; per-task pass counts; cost per trial and per solved task; the number of trials that ended at the step limit; trials with errors. Failed or errored trials count as failures; none are dropped or re-run.

## What this run cannot show

Five tasks from one repository, one model, no comparison with other agents or models. Task difficulty is not representative of general coding work. The model may have seen this public repository. A result here describes this suite, not coding agents in general.
