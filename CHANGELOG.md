# Changelog

All notable changes to Mazu are documented here, newest first. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); Mazu doesn't yet promise strict semver (see [ROADMAP.md](ROADMAP.md) for what "1.0" would mean) — treat version bumps as "a meaningful batch of work shipped," not a compatibility contract.

## 0.21.1 — Fix `mazu doctor` false "no API key set" for keys saved via `mazu setup`/`config set`

- **Fixed:** a real bug caught via live testing on a fresh install -- `mazu doctor` (and `mazu doctor --live`) reported "no provider has a key set" and FAILed even immediately after `mazu chat` had just worked correctly against a key saved via `mazu setup`. Root cause: `ensure_api_key()`/`default_model()` (every real code path) call `load_config()` first, which injects any `mazu config set`/`mazu setup`-saved key from `~/.mazu/config.toml` into the environment (only if the env var isn't already set) -- but `check_api_keys()` (doctor's underlying check) never called `load_config()`, so it only ever saw literal environment variables, never config-file-stored keys. `mazu doctor` now calls `load_config()` first too, matching every other real resolution path.

## 0.21.0 — `mazu explore --auto-models`: fill --models from the router's own history

- **Added:** `mazu explore TASK --approaches N --auto-models` -- picks the N models to compare automatically instead of requiring `--models` to be typed out by hand. Ranks candidates by this project's own router history for the task's (locally classified) type -- same win-rate-desc/cost-asc ordering `mazu router stats` already prints -- and tops up any remaining slots with one model per other available provider (detected the same way `mazu run`'s default-model auto-detection already works). Mutually exclusive with `--models`; fails with a clear error (not a silently repeated model) if fewer than N distinct models can be found, telling you to add provider keys, build more history first, or fall back to `--models`.
- This is the router closing its own loop one step further: `mazu explore` produces the history, `mazu router stats`/the passive `mazu run`/`mazu chat` suggestions surface it, and now `--auto-models` lets a *later* `mazu explore` comparison consume that same history to decide what to compare next, without the user needing to remember or retype which models have done well here before.
- **Fixed (caught by this addendum's own tests before shipping):** the first draft of the provider-topping-up logic double-prefixed the provider name (`deepseek:deepseek:deepseek-chat`) because `_PROVIDER_DEFAULT_MODELS`'s values already include the `provider:` prefix -- caught immediately by `test_cli_auto_models_wires_the_picked_list_into_run_explore` before this ever reached a real run.

## 0.20.0 — Learning model router: suggestions from your own project's `mazu explore` history

- **Added:** every `mazu explore` branch outcome (model, a local zero-cost task-type classification, cost via the existing `UsageStore`, and whether `--test-command` passed) is now persisted to the project's `.mazu/runs.db` instead of being discarded once the report prints. Two new nullable columns (`explore_group_id`, `test_passed`) added via the same additive-migration pattern as the existing lineage columns — raw facts only, never a stored "won" boolean, so ranking stays recomputable rather than frozen into history.
- **Added:** `mazu run`/`mazu chat`, when called with no `--model` and at least 3 matching data points exist in this project's history for a task of the same (locally classified) type, print a passive, clearly-labeled suggestion — e.g. `[router] DeepSeek has won most bug-fix explorations here (2/2, avg $0.0045) — consider --model deepseek:deepseek-chat.` followed by an explicit "never applied automatically" line. The suggestion is **only ever printed** — `model=None` is never reassigned anywhere in the router code path, and this is covered by a dedicated regression test that captures the real kwarg passed into `run_autonomous` and asserts it stays `None` even with strong seeded history pointing elsewhere.
- **Added:** `mazu router stats [--task-type ...] [--json]` — shows win rate, test-pass rate, and average cost per model, so the data behind any suggestion is always inspectable, never a black box.
- **Added:** `mazu config set router_suggestions false` to silence the suggestion print entirely (e.g. for scripting/CI use).
- Task-type classification (`bugfix`/`feature`/`refactor`/`test`/`docs`/`general`) is a deliberately simple, local, English-only keyword heuristic — not an LLM call, so a plain `mazu run`/`mazu chat` never pays extra latency or cost just to maybe print a suggestion. Project-scoped only, matching `RunStore`'s existing scope — never cross-project.
- **Live-verified** against real Anthropic and DeepSeek keys: ran multiple real `mazu explore` sessions against a scratch project, confirmed `mazu router stats` showed correct win-rate/cost aggregation matching `mazu runs`/`mazu usage` for the same session ids, and confirmed a plain `mazu run` (no `--model`) against matching history printed a real, correctly-labeled suggestion without altering which model actually ran.
- **Fixed (caught during live verification, not shipped broken):** each `mazu explore` branch runs in its own isolated `git worktree` with its own `.mazu/runs.db`, but router suggestions/stats are read from the *origin* project's own `.mazu/runs.db` — which never saw those rows, so `mazu router stats` silently showed "no history" even right after a real, successful explore run. Fixed by mirroring a summary row (including real final step/checkpoint counts, not just schema defaults) back into the origin project's `RunStore` after each explore run completes.

## 0.19.0 — `mazu explore`: parallel branch exploration, plus a real cross-process checkpoint race fixed

- **Added:** `mazu explore TASK --approaches N --models a,b,...` — forks N branches from one checkpoint into real, isolated `git worktree`s and runs the same task on each with a different model in parallel, then prints a ranked comparison (by test pass/fail if `--test-command` is given, otherwise by estimated cost). Nothing is ever auto-merged or auto-adopted — every branch's `cd <worktree>` (and, if you want it back in the main repo, `git worktree remove ... && git checkout ...`) is printed for you to run yourself. Strictly opt-in: never triggered by `mazu run`/`mazu chat`. `--allow-shell` is forced on for every branch (an unattended parallel branch can't answer an interactive confirmation prompt) and this is disclosed loudly every run, not buried in a flag. `--approaches` is hard-capped at 5 so a typo can't turn into an accidental large bill; `--max-cost` is shared across all branches via the same thread-safe tracker `mazu council` already uses, so one expensive branch can cut its siblings short.
- **Live-verified** against real Anthropic, DeepSeek, and OpenAI keys: real parallel 2-way and 3-way runs, each branch's file changes confirmed correct and isolated from the others and from the original working tree, `--test-command` correctly ran `pytest` per branch and ranked a passing branch above a cheaper failing one, and a tiny `--max-cost` correctly cut both branches short. One real bug found and fixed during this verification: a brand-new project with no checkpoint yet had nothing for `explore` to fork from — it now auto-creates a baseline checkpoint instead of requiring a separate manual step first.
- **Fixed:** a real cross-process race in checkpoint creation, caught live via CI (not just reasoned about) — two concurrent `mazu` processes committing to the same project could each read a stale index state and produce colliding or lost checkpoints. Two hand-rolled lock implementations (OS byte-range locks, then a directory+PID-staleness scheme) both looked correct and passed most local runs but were proven unreliable under real contention through extensive reproduction; replaced with `filelock`, the same small, widely-used library pip/virtualenv/tox rely on. Verified with 40/40 clean runs of a real two-process stress test.

## 0.18.0 — Local model support (LM Studio, Ollama, any OpenAI-compatible server)

- **Added:** a fifth provider, `local:<model-name>` — points Mazu at any OpenAI-compatible chat completions server running on your own machine (LM Studio, Ollama, llama.cpp's server), with no API key, no cloud egress, and no per-token cost. Reuses the same `openai` package as the `openai`/`deepseek` providers, so no new install extra is needed. Ollama-style multi-colon model tags work unchanged (`local:llama3.1:8b`).
- **Added:** `mazu config set local_base_url <url>` (and the `MAZU_LOCAL_BASE_URL` env var, which wins over the config value) to point at something other than LM Studio's default `http://localhost:1234/v1` — e.g. Ollama's `http://localhost:11434/v1`.
- **Added:** a new `requires_api_key` attribute on the `Provider` base class, so `config.ensure_api_key()` can skip its mandatory-key check for a provider that genuinely needs none, without hardcoding any provider name.
- `local` is deliberately excluded from provider auto-detection — a reachable local server is never enough on its own to make Mazu silently switch to it; it must always be chosen explicitly via `--model`, `MAZU_MODEL`, or `mazu config set default_model`.
- Live-verified end-to-end against a real running LM Studio instance (`qwen/qwen3.5-9b`): `mazu run --model local:qwen/qwen3.5-9b "..."` completed a full autonomous tool-use round-trip (file creation, checkpointing) with no auth errors.
- Context window and pricing correctly show as unknown (`?`) in `mazu models` for local models — genuinely unknowable for an arbitrary user-loaded model, not fabricated as `$0`.

## 0.17.0 — Detect long-running server commands before they block `run_shell`

- **Added:** `run_shell` now recognizes commands that start a long-running dev server/process that never exits on its own (`flask run`, `npm run dev`/`start`, `yarn dev`/`start`, bare `vite`, `next dev`, `uvicorn`/`gunicorn`, `python -m http.server`, `manage.py runserver`, `rails server`) and refuses to run them, returning an immediate, clear message instead of blocking for the full timeout. Found via a real live session: the model tried `python app.py` to "test" a Flask site it had just built, which blocked until the user manually interrupted the whole `mazu chat` process with Ctrl+C.
- **Added:** for the common case that motivated this — a bare `python <file>.py` invocation, which gives no hint on the command line itself about what the script does — `run_shell` also peeks at the target script's own source for well-known server-starting calls (`app.run(`, `.run(host=`, `.run(debug=`, `serve_forever(`, `uvicorn.run(`) before running it. Deliberately narrow signatures (not a bare `.run(` search) to avoid false-positiving on legitimate scripts that happen to call `subprocess.run(...)` or similar.
- Compound commands (`cd sub && python app.py`) and scripts with extra arguments are intentionally not source-inspected — an accepted false negative, since reliably parsing an arbitrary shell command for its target file isn't worth the complexity; these still fall through to the existing timeout behavior, unchanged from before.

## 0.16.2 — Fix Windows subprocess crashes on non-ASCII output

- **Fixed:** a real bug found via live testing, observed twice independently (once with an emoji in a generated print statement, once with a Turkish/Azerbaijani letter in a saved skill) — on Windows, a subprocess spawned by `run_shell` or a saved skill defaults to the console's legacy codepage for its own stdout, not UTF-8, so printing non-ASCII text crashed the *subprocess itself* with `UnicodeEncodeError`. The model would then have to notice and fix this in an extra round, wasting a step and tokens every time. Both `mazu/tools/shell.py::run_shell` and `mazu/skills/manager.py::SkillManager.run` now set `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` for the subprocess environment, and capture stdout/stderr with explicit `encoding="utf-8", errors="replace"` (previously relied on `text=True`'s locale-dependent decoding).

## 0.16.1 — Fix misleading `mazu timeline` message

- **Fixed:** a real bug found via live testing (a real `mazu run --allow-shell` session), not a hypothetical one — `mazu timeline` printed "(first checkpoint — nothing to compare against)" for *any* checkpoint with no changed files, even when it had a real parent and that round simply didn't touch any tracked file (e.g. a read-only/inspection-only tool round). Now distinguishes "true root, nothing to diff against" from "a real predecessor exists, but nothing changed this round" — confirmed live: an 8-checkpoint autonomous run showed steps 7 and 8 as false "first checkpoints" before this fix.

## 0.16.0 — Machine-readable output (`--json`)

- **Added:** `--json` on the five read-oriented commands ROADMAP's Phase M names as what an integration would actually call: `mazu timeline`, `mazu memory list`, `mazu log`/`mazu log show`, `mazu runs`, `mazu models`. Every payload is wrapped in a versioned envelope (`{"schema_version": 1, "mazu_version": ..., "data": ...}`) via a new small `mazu/output.py` module, so a consumer can check compatibility instead of silently misparsing a future change.
- **Added:** `mazu runs --json` is the first time the run-lineage columns added for branching checkpoints (`origin_checkpoint_id`, `parent_run_id`, `branch_name`) become visible through the CLI in any form — the text output never printed them.
- **Fixed (caught during implementation, not shipped):** the initial plan assumed `mazu memory list`'s underlying query never selects the `embedding` column; it does (`SELECT *`). Explicitly excluded it from `--json` output rather than leaking an internal representation detail, verified live before merging.
- **Note:** structured (non-`--stat`) diff data for `mazu checkpoint diff`/`compare`/`preview_rollback` remains deliberately deferred — a separate, later piece of work per ROADMAP's own scoping.

## 0.15.0 — Cost-efficiency hardening + crash-safe writes

- **Added:** multi-breakpoint Anthropic prompt caching — the tool schema list and the last message of the conversation now get `cache_control` breakpoints too, not just the system prompt. Live-verified against a real multi-step `mazu run`: after the first step, each subsequent step showed only ~2 fresh input tokens with 3,900+ read from cache, instead of re-processing the whole growing tool/message set every step. `run_forced_tool` (used by memory extraction and context compaction) also gained system-prompt caching, which it was previously missing entirely.
- **Added:** `openai:gpt-4o`, `openai:gpt-4o-mini`, and `gemini:gemini-2.0-flash` to the pricing table — `--max-cost`, `mazu usage`, and `mazu models` were previously silently blind to these models (best-effort rates, flagged for verification against provider docs).
- **Added:** `mazu council --max-cost <usd>` — a thread-safe shared spend cap across all council members (previously council mode had zero cost guardrail of any kind, unlike `mazu run`). When the budget is exhausted, members stop taking further rounds and the lead synthesis call is skipped with a clear message instead of firing a full-price call over budget. Always-on `[cost]` line now shown at the start of every council run, regardless of whether `--max-cost` is set.
- **Added:** crash-safe (atomic) writes for `write_file`/`edit_file` — both now write to a temp file in the same directory, fsync it, then atomically rename it onto the real path. If the process or machine dies mid-write, the original file is left fully intact rather than truncated or corrupted.
- **Fixed:** a real bug found via live testing — council members that answered a question directly without ever using a tool (the common case for simple questions) skipped the cost-tracking check entirely, silently letting `--max-cost` do nothing for that member. The check now runs for every round unconditionally; only the decision to take another round is gated by budget exhaustion.

## 0.14.0 — Branching checkpoints

- **Added:** `mazu run --from-checkpoint <id> --branch <name> "<task>"` — forks a new, divergent line of execution from any existing checkpoint (git branch + `memory.db` + `.mazu/skills/` restored onto it, brand-new run/session id) and runs a task on it. Unlike `--resume`, this is not a continuation: the origin branch's later checkpoints are left completely untouched. Built on `CheckpointManager.branch_from()` (existing, git-only) plus a new `fork()` that adds the state restore, reusing `restore()`'s own logic rather than duplicating it.
- **Added:** `mazu checkpoint compare-branches <run_id_a> <run_id_b>` — status/steps/stop-reason/memories-saved/estimated-cost for two runs side by side, plus a real diff between their final checkpointed states and their common ancestor checkpoint (if the two share one).
- **Fixed:** three places that silently assumed checkpoint history was one linear chain, which only breaks once a divergent branch actually exists — caught and fixed as part of this addition rather than discovered later: `mazu checkpoint prune` could delete a divergent branch's only checkpoints just because the main line produced more newer ones in the meantime; `mazu timeline` diffed a checkpoint against whichever entry happened to sit next to it in the flat index instead of its real git parent; a no-argument `mazu rollback` could target a checkpoint actually made on a different branch. All three are fixed via two new additive index-entry fields (`branch`, `parent_checkpoint_id`) that older checkpoints simply lack — existing single-branch projects see byte-identical output.

## 0.13.1 — Launch readiness

- **Added:** this file.
- **Added:** `ROADMAP.md` Phase M (plan only, not implemented) — `--json` output for the read-oriented commands, and structured (not just `--stat`) checkpoint diffs. Surfaced by the VS Code extension design doc (0.12.1) as a real prerequisite for any editor integration.
- **Changed:** README — status badges, an "alpha, but usable" line, real screenshots of `mazu ui`, and an expanded Safety model section covering the shell allowlist, `--dry-run`, and config-key masking added since it was last written.

## 0.13.0 — Terminal UI

- **Added:** `mazu ui` — an interactive, full-screen terminal UI (built with [Textual](https://github.com/Textualize/textual), optional `mazu[ui]` extra) over the same data `mazu timeline`/`mazu memory list`/`mazu log` expose. Three tabs: Checkpoints (`r` to roll back), Memory (`p` to pin/unpin), Action Log (select a session to drill into its tool calls). Every mutation goes through the same store methods the CLI commands already use.

## 0.12.2 — Demo walkthrough

- **Added:** [docs/demo.md](docs/demo.md) — a real, reproducible walkthrough of the scenario checkpoints and memory exist for: an agent makes a plausible mistake, `mazu timeline`/`checkpoint diff`/`log show` find and explain it, `mazu rollback` undoes exactly that checkpoint, and a `mistake`-category memory makes sure it isn't repeated.

## 0.12.1 — VS Code/Cursor extension design doc

- **Added:** [docs/vscode-extension-design.md](docs/vscode-extension-design.md) — a design-only document for a possible future editor extension. Not implemented; the extension would be a separate project. Identified a real prerequisite: Mazu has no machine-readable output today (see 0.13.x's Phase M plan below).

## 0.12.0 — Onboarding

- **Added:** `mazu doctor --fix` — auto-fixes what's safely fixable without a value only the user has (a missing `.gitignore` entry, an uninitialized git repo).
- **Added:** `mazu setup` — a guided first-run wizard: pick a provider, paste an API key (persisted to `~/.mazu/config.toml`, masked wherever it's shown), optionally verify it live, optionally set it as `default_model`, optionally initialize the current directory.

## 0.11.0 — Provider layer (2/3)

- **Added:** `mazu models` — a capability table (real streaming support, tool use, approximate context window, approximate pricing) across every known provider/model, backed by `mazu/llm/capabilities.py`.
- **Added:** `mazu config set/list/unset` — persistent settings in `~/.mazu/config.toml` (`default_model`, per-provider API keys), generalized from a single hardcoded Anthropic-only field. Environment variables always take priority.
- **Deferred (documented, not silently dropped):** real streaming for Gemini. The `google-genai` SDK's docs/source don't confirm whether function-call parts arrive fragmented, whole, or only in the final chunk during a stream; guessing wrong risks silently mishandling tool calls. Needs a live Gemini key with real quota to verify first.

## 0.10.0 — Resumable runs

- **Added:** a formal run-id system (`RunStore`, `.mazu/runs.db`) — `mazu runs` lists every `mazu run` invocation with its config, status, and stop reason.
- **Added:** `mazu run --resume <run_id>` — continues a run that stopped short (hit `--max-steps`, `--max-cost`, or was interrupted) from its last checkpoint's conversation state, reusing the original task/model/options.
- **Added:** a structured end-of-run report (files changed, checkpoints created, memories saved, tool errors) printed at the end of every `mazu run`, fresh or resumed.

## 0.9.0 — Safer execution

- **Added:** `mazu run --dry-run` — runs the full task loop for real, but `write_file`/`edit_file`/`run_shell` report what they'd do instead of touching disk or a shell. Read-only tools still run for real.
- **Added:** `--shell-allowlist` (both `mazu chat` and `mazu run`) — restricts shell commands to a named set of programs, opt-in and additive to the existing denylist (which always applies on top, regardless).
- **Changed:** denylist block messages now say *why* a command was blocked ("elevates privileges via sudo") instead of a generic "matches the safety denylist."

## 0.8.0 — Agent action log

- **Added:** a persistent log of every tool call an agent session makes — tool name, input, outcome (ok/error/blocked/declined/unknown_tool), output summary, changed file — in `.mazu/action_log.db`. `mazu log` lists recent sessions; `mazu log show <session_id>` shows the full detail.

## 0.7.0 — Memory UX

- **Added:** `mazu memory why "<query>"` — shows exactly which memories would be retrieved for a query and why (BM25/semantic score, or the always-on `pinned`/recent-`mistake` floor), reusing the real selection logic rather than approximating it.
- **Added:** `mazu memory pin/unpin`, `mazu memory edit`, `mazu memory supersede`, `mazu memory stats`.
- **Added:** `retrieval_count`/`last_used_at` tracking per memory, shown in `mazu memory list`.

## 0.6.0 — Checkpoint UX (part 2)

- **Added:** `mazu checkpoint inspect <id> --memory/--conversation` — reads a checkpoint's frozen snapshot directly, without restoring it.
- **Added:** `mazu checkpoint compare <id1> <id2>` — diffs two checkpoints directly, not either against the current state.
- **Added:** `mazu branch-from <id> <branch-name>` — creates a git branch at a checkpoint's commit; non-destructive, doesn't touch the current branch or working tree.

## 0.5.1 — Phase A: identity & cleanup

- **Changed:** README repositioned around Mazu's actual pitch ("a local memory and checkpoint layer for coding agents," not a Cursor/Claude Code clone).
- **Added:** [ARCHITECTURE.md](ARCHITECTURE.md) — the system-design detail split out of README.

## 0.5.0 — Checkpoint UX (part 1)

- **Added:** `mazu timeline`, `mazu checkpoint show <id>`, `mazu checkpoint diff <id>` — see checkpoint history and what changed without a raw `git log`.

## 0.4.0 — Semantic memory search

- **Added:** optional semantic memory search (`MAZU_SEMANTIC_MEMORY=1`) — blends BM25 keyword ranking with embedding cosine similarity (50/50) so a memory phrased very differently from the current task can still surface. Opt-in; off by default.

## 0.3.0

- **Added:** Gemini provider (`gemini:*` models).
- **Added:** `mazu doctor` — diagnoses common setup problems (Python/git availability, which provider keys are configured, project readiness), with an optional `--live` real-API-call check.
- **Added:** `mazu memory consolidate` — finds near-duplicate memories and merges them (marks superseded, deletes nothing) via fuzzy title/body similarity.
- **Fixed:** missing-API-key error message showed bash `export` syntax even on Windows.

## 0.2.0

- **Added:** live cost display during `mazu chat`/`mazu run`, plus persistent `mazu usage` (cross-project spend tracking, approximate pricing table).
- **Added:** real-time token-by-token streaming in `mazu chat`.
- **Added:** context compaction in `mazu run` for long-running autonomous tasks.
- **Added:** pytest test suite + GitHub Actions CI (Python 3.11–3.13, Linux/Windows/macOS).
- **Added:** CONTRIBUTING.md, SECURITY.md, `mazu --version`, `mazu checkpoint list`.

## 0.1.2

- **Changed:** strengthened the system prompt — identity, version, and capability/privacy disclosure.

## 0.1.1

- **Fixed:** a crash when the `openai` package wasn't installed but an `openai:*`/`deepseek:*` model was requested.

## 0.1.0

- Initial release: multi-provider chat/autonomous-run loop (Anthropic, OpenAI, DeepSeek), project-scoped persistent memory (`.mazu/memory.db`), and checkpoint/rollback (`mazu checkpoint`, `mazu rollback`) built on git.
