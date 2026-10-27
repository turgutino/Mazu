# Mazu

[![PyPI](https://img.shields.io/pypi/v/mazu)](https://pypi.org/project/mazu/) [![CI](https://github.com/turgutino/Mazu/actions/workflows/ci.yml/badge.svg)](https://github.com/turgutino/Mazu/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

**A local coding agent with persistent project memory, rollback, cost limits, and a built-in way to measure itself.**

Mazu is alpha software. It runs entirely on your machine, works with Anthropic, OpenAI, DeepSeek, Gemini, or a local OpenAI-compatible server, and keeps its own state in a `.mazu/` folder inside your project. This README tries to say plainly what works, what has been measured, and what has not — see [Measured so far](#measured-so-far) and [Known limitations](#known-limitations).

```
  ███╗   ███╗
  ████╗ ████║   A Z U
  ██╔████╔██║   ══════════════════════════════
  ██║╚██╔╝██║   memory-augmented coding agent
  ██║ ╚═╝ ██║   persistent memory · checkpoints · skills · multi-model
  ╚═╝     ╚═╝
```

## What Mazu does

- **Chat or run unattended.** `mazu chat` is an interactive agent that can read and edit files and run shell commands, asking before anything destructive. `mazu run "task"` does the same in a loop without you, with step and cost limits.
- **Remembers a project across sessions.** Decisions, conventions, and mistakes are stored in a local SQLite database, ranked against your current task with local BM25 (optionally blended with embeddings), and injected back into context. Facts about *you* live in a separate global store that follows you into every project.
- **Checkpoints code, memory, and conversation together.** Each step of a run snapshots a git commit, a consistent copy of the memory database, the skill library, and the transcript. `mazu rollback` restores them; `mazu timeline` and `mazu checkpoint diff` show what changed first.
- **Bounds spend.** `--max-cost` on `mazu chat` and `mazu run` counts the whole session's estimated spend — the main loop, context compaction, and memory extraction — and `mazu usage` totals it across projects.
- **Measures itself.** `mazu bench` replays a fixed task suite several times under different configurations and reports pass rate with confidence intervals and cost per solved task, so a change to the agent can be judged on numbers instead of impressions.
- **Tries more than one way.** `mazu explore` runs the same task on several models in parallel in isolated git worktrees and ranks the results (optionally by your test suite); `mazu council` asks several models the same read-only question and has a lead model synthesize an answer.

Mazu is not an IDE and does not try to out-polish editor-integrated agents. Its focus is the layer around the model: what the agent remembers, what it did, whether you can undo it, and what it cost.

## Measured so far

Mazu ships with benchmarks built from its own code, judged only by hidden tests that the agent cannot see or edit (they are kept outside the clone and copied in after the agent finishes). `--check` on each suite proves every task is fair without calling a model: the unfixed code must fail its test and the fix must pass it. Full numbers, caveats, and how to reproduce them are in [bench/RESULTS.md](bench/RESULTS.md).

All numbers below are `deepseek:deepseek-chat`, plain `mazu run`, 95% Wilson intervals:

| Suite | What it is | Pass rate | Cost per solved task |
|---|---|---|---|
| [`bench/mazu_history`](bench/mazu_history) | 5 real bugs from this repository's own history, each checked out at the commit before its fix, described by symptom only | 25 / 35 = 71% (55%-84%) | about $0.051 |
| [`bench/mutation_bugs`](bench/mutation_bugs) | 9 one-line bugs injected into Mazu's current code | 27 / 27 = 100% (88%-100%) | about $0.022 |

Per task, on the real-bug suite: three tasks passed almost every time, one (`windows-non-ascii-output`) passed 6/7, and one, `restore-open-connection` (an open SQLite connection makes a rollback silently not revert the database), passed **0 of 22** trials across the pilot and confirmation runs, with either configuration. The injected-bug suite is a ceiling: everything passes, so it cannot tell two configurations apart.

An experimental prompt rule (`mazu run --protect-tests`, "do not edit existing tests") looked helpful in a first small run (19/25 against 10/15) and did not hold up in a confirmation run whose analysis was fixed in advance ([bench/analysis_plan.md](bench/analysis_plan.md)): 24/35 against 25/35 on the real bugs and 24/27 against 27/27 on the injected ones. It is off by default and not shown to help.

What this does **not** show: the suites are small, come from one repository the model may have seen, and use one model. Mazu has not been compared head-to-head with any other coding agent or with a more expensive model on the same tasks. Treat the numbers as a measured baseline for judging changes, not as a ranking.

## Install

Requires **Python 3.11+** and **git** (used for checkpoints).

```bash
pip install mazu
pip install "mazu[openai]"      # openai:* and deepseek:* models
pip install "mazu[gemini]"      # gemini:* models
pip install "mazu[ui]"          # terminal UI (mazu ui)
pip install "mazu[documents]"   # .docx/.pdf/.xlsx/.pptx extraction (used by mazu-web)
```

From source: `git clone https://github.com/turgutino/Mazu.git && cd Mazu && pip install -e ".[openai]"`.

Set at least one provider key (Mazu uses whichever is present), or run the wizard:

```bash
export DEEPSEEK_API_KEY=...      # or ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
mazu setup                       # guided: pick a provider, store the key in ~/.mazu/config.toml
mazu doctor                      # diagnose setup problems (add --live to test keys, --fix for safe fixes)
```

## Quick start

```bash
cd your-project/
mazu init
mazu chat
```

```
> we use PostgreSQL for this project, not SQLite. remember that.
[remember] saved (decision): use PostgreSQL, not SQLite
> /checkpoint
[checkpoint] cp_000001 saved (commit 4f2a91c)
```

In a later session the agent already has that decision in context. For an end-to-end example — an agent breaks a test, `mazu timeline` and `mazu checkpoint diff` show what, `mazu rollback` undoes it — see [docs/demo.md](docs/demo.md).

In `mazu chat`: `/checkpoint` snapshots now, `/rollback [id]` restores, `/compact` compacts the conversation, `/auto` toggles auto-approval of destructive tools.

## Commands

### Agent

| Command | Purpose |
|---|---|
| `mazu chat` | Interactive session. `--model`, `--max-cost USD`, `--shell-allowlist`, `--auto-approve`, `--resume ID`, `--list`, `--delete ID` |
| `mazu run "task"` | Unattended multi-step run. `--max-steps` (default 15), `--max-cost USD`, `--allow-shell`, `--shell-allowlist`, `--dry-run`, `--checkpoint-every`, `--keep-checkpoints`, `--model`, `--auto-commit-baseline`, `--protect-tests` (experimental: tells the agent not to edit existing tests; a pre-registered test found no benefit) |
| `mazu run --resume ID` | Continue a stopped run from its last checkpoint |
| `mazu run --from-checkpoint CP --branch NAME "task"` | Start a divergent run on a new git branch from a checkpoint |
| `mazu runs` | List recent runs: status, stop reason, steps |
| `mazu explore "task"` | Same task on N models in parallel git worktrees, ranked (`--models` or `--auto-models`, `--test-command`, `--max-cost` shared across branches). Never auto-merges. |
| `mazu council "question"` | Ask several models the same read-only question; a lead model synthesizes |

`mazu run` refuses to start on a dirty working tree (unless `--auto-commit-baseline` or `--dry-run`) so the first checkpoint is a clean baseline. `--dry-run` runs the model for real but reports file writes and shell commands instead of executing them.

### Checkpoints

```bash
mazu checkpoint                   # snapshot now
mazu checkpoint list | show ID | diff ID | inspect ID --memory | compare A B | prune --keep 20
mazu checkpoint compare-branches RUN_A RUN_B
mazu timeline                     # what changed at each checkpoint
mazu rollback [ID]                # restore code + memory + skills + conversation
mazu branch-from ID NAME          # new git branch at a checkpoint, nothing else touched
```

A checkpoint is a git commit, an online-backup copy of `memory.db`, a copy of the skills folder, and the conversation. See [Known limitations](#known-limitations) for how restore behaves if it is interrupted.

### Memory

```bash
mazu memory list [--category C] [--global]
mazu memory why "query"           # what would be retrieved for this query, and why
mazu memory pin ID | unpin ID | edit ID | forget ID | supersede OLD NEW
mazu memory consolidate --dry-run # preview near-duplicate merging, then run without --dry-run
mazu memory stale [--auto]        # find (or archive) memories that look stale
mazu memory stats
```

Categories: `decision`, `convention`, `mistake`, `task_outcome`, `fact` (project) and `user_preference` (global). Memories are written two ways: explicitly when the agent calls `remember`, and by a cheap end-of-session extraction pass (at most 5 per session, de-duplicated against existing entries by title and word overlap). At most 15 memories and about 8,000 characters are injected per session; pinned memories and the three most recent mistakes are always included.

Optional semantic search (`MAZU_SEMANTIC_MEMORY=1`, needs an OpenAI key for embeddings) blends BM25 with cosine similarity 50/50; off by default.

`mazu memory-belief stats | review` exposes an observation-only log of memory-contradiction decisions. It never changes memory.

### Skills, router, curator

- **Skills** (`mazu skills list | forget | unarchive`): the agent can save a reusable Python function and run it later without a model call. See the safety notes below before relying on this.
- **Router** (`mazu router stats`): after `mazu explore` runs, Mazu can print a passive model suggestion based on your own project's results. It never changes which model runs. Disable with `mazu config set router_suggestions false`.
- **Curator** (`mazu curator setup | run | status | log | undo | enable | disable`): an opt-in maintenance pass over Mazu's own memory and skills using its own separate API key. Inert until configured; every change is logged and reversible.

### Cost, logs, diagnostics

```bash
mazu usage [--since-days 7]       # estimated spend by model, all projects (~/.mazu/usage.db)
mazu models                       # known models: streaming, tools, context window, approximate price
mazu log | log show SESSION       # every tool call the agent made, including blocked and declined ones
mazu doctor [--live] [--fix]
mazu config set|unset|list       # ~/.mazu/config.toml; environment variables always win
mazu ui                           # terminal UI over checkpoints, memory, and the action log
```

`mazu timeline`, `memory list`, `log`, `runs`, and `models` accept `--json`. Costs are estimates from a built-in price table ([`mazu/llm/pricing.py`](mazu/llm/pricing.py)) that can lag behind provider pricing; a model without a known price is reported as unpriced rather than counted as free, and `--max-cost` cannot be enforced for it.

### Benchmarking your own changes

```bash
python bench/mazu_history/build_suite.py --out ./bench-work/mazu-history --check   # no model calls
mazu bench run --suite bench-work/mazu-history/suite.json --configs configs.json --repeats 3 --out results.jsonl
mazu bench report results.jsonl
```

`configs.json` is a list such as `[{"name": "baseline", "model": "deepseek:deepseek-chat", "extra_args": ["--max-cost", "0.60"]}]`. Each trial runs in a fresh `git clone` with a throwaway home directory (your real `~/.mazu` is untouched; API-key config is copied in), the agent runs with shell commands allowed inside that clone, and pass/fail comes from the task's verify command, never from the agent's own claim. `mazu bench run` makes real, billed model calls. Results are an append-only file, so an interrupted run resumes. Configs are interleaved per task so provider drift affects all of them alike. The report shows pass rate with a Wilson 95% interval and says when intervals overlap — with few repeats an apparent gap is usually not evidence of a real difference.

A second, easier suite of eight seeded bugs is in [`bench/seeded_bugs`](bench/seeded_bugs).

## Models

Models are named `provider:model`, for example `deepseek:deepseek-chat`, `openai:gpt-5`, `anthropic:claude-sonnet-5`, `gemini:gemini-2.0-flash`. Without `--model`, Mazu uses `MAZU_MODEL`, then `default_model` from `mazu config`, then whichever provider key is set. `mazu models` lists what Mazu knows about; names and prices there can go stale, and models newer than the table appear as unpriced.

`local:<model-name>` points at any OpenAI-compatible server on your machine (LM Studio by default at `http://localhost:1234/v1`; set `local_base_url` for Ollama or others). It is never auto-selected. Smaller local models are noticeably weaker on multi-file tasks.

`mazu config set reasoning_effort low|medium|high` controls reasoning effort for providers that support it; others accept and ignore it.

## Safety model

What Mazu does:

- File tools are confined to the project root, including through symlinks.
- Destructive tool calls ask for confirmation in `mazu chat`. In `mazu run`, shell commands ask unless `--allow-shell` is passed; file edits proceed because checkpoints make them recoverable.
- A hardcoded denylist blocks a short list of patterns: `rm -rf /`, `git push --force`, paths under `.ssh/`, `sudo`, and `format X:`. Every block says why.
- `--dry-run` previews a task without writing or executing.
- API keys are read from the environment or `~/.mazu/config.toml` (in your home folder, outside any project) and shown masked. Mazu adds `.mazu/` to your project's `.gitignore`; it does not touch `.env` files, so keep those out of version control yourself.

What it does **not** do — read this before using `--allow-shell` or unattended runs:

- **The denylist is a backstop, not a sandbox.** It matches a few exact patterns and is easily bypassed by equivalent commands (for example `rm -rf ~`, `git push -f`, `su`, PowerShell deletes, or a Python one-liner). Run unattended agents in a disposable directory or container.
- **`--shell-allowlist` only checks how a command starts.** A compound command such as `git status; something-else` passes an allowlist of `git`.
- **Skills are arbitrary Python** run as a subprocess with Mazu's privileges and environment, with no isolation. In `mazu run`, `--allow-shell` governs `run_shell` only; saved skills are executed without a confirmation prompt (in `mazu chat` they do ask). This is a known gap.
- Checkpoints undo file and memory changes. They cannot undo an external action such as a push or a network request.

See [SECURITY.md](SECURITY.md) for the policy and how to report a vulnerability.

## Known limitations

- **No verification step.** The agent decides for itself when it is done, and nothing re-runs your tests before it stops. In the baseline above, an agent twice edited an existing test to fit its change.
- **Rollback is not atomic.** `restore` resets git, then restores memory, skills, and the index in sequence. If it is interrupted midway, code may be rolled back while memory or skills are not (4 of its 7 steps leave that mixed state when interrupted), but running `mazu rollback` again repairs it. This is tested by making each step fail in turn; a process killed in the middle of writing a file has not been tested.
- **Memory is not verified.** Auto-extracted memories are injected into the prompt as established context, with no trust level. Contradictions between memories are detected in shadow mode only (logged, never acted on), so a wrong remembered fact can persist until you edit or forget it. Auto-extraction keeps short summaries, not the raw conversation. Memory quality has not been benchmarked.
- **Memory growth is manual to clean up.** At most 5 memories are added per session and the prompt block is bounded, but the database itself only shrinks through `consolidate`, `stale`, `forget`, or Curator.
- **Small evidence base.** The benchmark covers five bugs from one repository, with one model, and no comparison against other agents. Context compaction and memory extraction are not exercised by those short tasks.
- **Platforms.** Live testing has mostly been on Windows. CI runs the test suite (about 1,100 tests, no API cost by default) on Linux, Windows, and macOS across Python 3.11–3.13. Gemini is implemented but has not been exercised in recent live tests and does not stream.
- Rollback is linear; branching exists (`--from-checkpoint`, `compare-branches`) but there is no merge or graph view.

The build plan is in [ROADMAP.md](ROADMAP.md), version history in [CHANGELOG.md](CHANGELOG.md), and module-level design in [ARCHITECTURE.md](ARCHITECTURE.md).

## Web UI

[`mazu-web`](https://github.com/turgutino/mazu-web) is a separate companion package with the same operations in a browser (chat, run, explore, council, checkpoints, memory, skills, usage, doctor, config), plus image and document attachments.

```bash
pip install mazu-web
cd your-project && mazu-web      # http://127.0.0.1:8765
```

Its chat tab keeps its own copy of the chat loop, so it does not yet have `mazu chat --max-cost` or the final session total described above; its Run, Explore, and Council tabs share the same code as the terminal.

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Turgut Sofuyev.
