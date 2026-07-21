# Mazu Roadmap

**Pitch:** Mazu is a local memory and checkpoint layer for coding agents — not a Cursor/Claude Code replacement. It exists to make an agent's history (what it decided, what it changed, what it remembers) visible, reversible, and trustworthy.

This file is split into two parts on purpose:
- **Committed** — the full build sequence, in order. Each phase becomes real code with tests before we call it done and move to the next.
- **Directional (not committed)** — ideas worth keeping on record, deliberately not started yet because they need a real second audience (a team, other contributors, real users) to be worth the complexity. Not a promise, not a deadline — revisit if that changes.

## Committed — full build sequence

### Phase A — Identity & cleanup ✅ done
- [x] `ROADMAP.md` exists
- [x] Test suite + CI green
- [x] README repositioned around the actual pitch instead of "coding agent CLI" (What is Mazu / Why not Cursor / Core idea structure)
- [x] `ARCHITECTURE.md` — the system-design detail split out of README, which stays focused on usage
- [x] Clean up stray untracked files in the repo root

### Phase B — Checkpoint UX ✅ done
- [x] `mazu timeline`
- [x] `mazu checkpoint show <id>`
- [x] `mazu checkpoint diff <id>`
- [x] `mazu checkpoint inspect <id> --memory` / `--conversation` — prints a snapshot's captured memory rows / conversation transcript directly, from the frozen snapshot, without restoring
- [x] `mazu checkpoint compare <id1> <id2>` — diff between two checkpoints directly (not checkpoint-vs-current)
- [x] `mazu branch-from <id> <branch-name>` — creates a git branch from a checkpoint's commit; deliberately git-only (no memory/skills restore) and non-destructive (current branch/working tree untouched)

### Phase C — Memory UX ✅ done
- [x] `mazu memory why "<query>"` — shows which memories would be retrieved and why (BM25 score, semantic score if enabled)
- [x] `mazu memory pin <id>` / `unpin <id>` — the `pinned` column already exists; just needs a command
- [x] `mazu memory stats` — counts by category/source, oldest/newest, superseded count
- [x] `mazu memory edit <id>` — edit an existing memory's title/body in place
- [x] `mazu memory supersede <old_id> <new_id>` — CLI wrapper around the `supersede()` method `remember` already uses internally
- [x] Track `last_used_at` / `retrieval_count` per memory (new columns + update-on-retrieval), shown in `mazu memory list`

### Phase D — Agent Action Log ✅ done
- [x] Persistent log of every tool call: time, tool name, input, output summary, changed files, outcome (ok/error/blocked/declined/unknown_tool) — `.mazu/action_log.db`, project-scoped like memory.db
- [x] `mazu log` — list recent sessions/runs
- [x] `mazu log show <session_id>` — full action log for one session

### Phase E — Safer Execution ✅ done
- [x] `mazu run "..." --dry-run` — show what would happen without writing/running anything; skips the clean-working-tree gate and checkpoint creation, since nothing changes; read-only tools still run for real
- [x] Shell command allowlist mode (opt-in, alongside the existing denylist) — `--shell-allowlist` on both `mazu chat` and `mazu run`; the denylist is a hard backstop that always applies on top, an allowlist entry can't rescue a denylisted command
- [x] Clearer explanation text when a command is blocked — each denylist rule now carries a human-readable reason ("elevates privileges via sudo") instead of a generic "matches the safety denylist"

### Phase F — Better Autonomous Runs ✅ done
- [x] Run id system — new `RunStore` (`.mazu/runs.db`) formalizes `session_id` as a first-class "run": config, live progress, status, stop reason; `mazu runs` lists them
- [x] `mazu run --resume <run_id>` (adapted from `mazu run resume <run_id>` — Click's Group/positional-argument handling makes a `resume` subcommand ambiguous against a free-text TASK argument; a flag on the existing `run` command is unambiguous and equally discoverable via `--help`). Resumes from the run's last checkpoint, reusing its original task/model/options exactly
- [x] Structured end-of-run report: files changed, checkpoints created, memories saved, tool errors — printed at the end of every `mazu run` (fresh or resumed), computed from `ActionLogStore`/`RunStore` rather than duplicated bookkeeping

### Phase G — Provider Layer ✅ done (2/3 — see note)
- [x] Model capability table (streaming support, tool support, context window, approximate cost) — new `mazu models` command, backed by `mazu/llm/capabilities.py`
- [x] `mazu config set / list / unset` for persistent settings (default model, per-provider API keys) — `~/.mazu/config.toml`, generalized from the old Anthropic-only `api_key` field; env vars still always win
- [ ] Real streaming for Gemini — **deliberately deferred**, not implemented. The `google-genai` SDK's `generate_content_stream` docs/source don't confirm whether `function_call` parts arrive fragmented, whole, or only in the final chunk during a stream (unlike Anthropic's `stream()`/`get_final_message()` pattern or OpenAI's well-documented delta-accumulation, both already verified and implemented). Guessing wrong risks silently mishandling tool calls for every Gemini user. Needs a live Gemini key with real quota (the one used earlier in this project had zero free-tier quota) to verify chunk behavior before implementing — revisit then.

### Phase H — Install & Onboarding ✅ done
- [x] `mazu doctor --fix` — auto-fixes what's safely fixable with no judgment call needed (missing `.gitignore` entry, uninitialized git repo). Deliberately leaves API keys/Python version/package installs as report-only, since those need a value or a decision only the user has
- [x] `mazu setup` — first-run wizard: pick a provider, paste a key (persisted to `~/.mazu/config.toml`, masked everywhere it's shown), optionally verify it live, optionally set it as `default_model`, optionally initialize the current directory

### Phase I — Documentation overhaul
- [ ] README restructured: what is Mazu / why not Cursor / core idea / quickstart / checkpoint demo / memory demo / safety model / architecture (linking out to `ARCHITECTURE.md`) / roadmap
- [ ] `examples/` — at least one worked-through example project
- [ ] `docs/` if the above grows past what fits comfortably in a few root-level files

### Phase J — Demo ✅ done
- [x] A concrete, scripted demo scenario (agent makes a mistake → `mazu timeline` finds the bad step → `mazu checkpoint diff` shows the damage → recover via rollback → memory means it isn't repeated) — [docs/demo.md](../docs/demo.md). Every command/output shown is real (run against an actual project, actual git, actual SQLite stores, twice — once to write it, once fresh from scratch to verify reproducibility); only the model's own responses are scripted/deterministic rather than a live API call, so the walkthrough doesn't depend on what a live model happens to say on a given run. A recorded video/GIF is still the user's own call, not a code task.

### Phase K — Terminal UI (`mazu ui`) ✅ done
- [x] A TUI over checkpoint timeline / memory browser / action log, built with `textual` (new optional `mazu[ui]` extra). Three tabs; rollback and pin/unpin are wired to the same `CheckpointManager`/`MemoryStore` methods the CLI commands use, not a second implementation. Verified two ways: Textual's own `Pilot`-driven test harness (real key presses/clicks against a real temp project, real git, real SQLite — proving every mutation actually happens, not just that a screen renders) and a rendered screenshot of all three tabs (including the action-log drill-down) reviewed visually before considering this done.

### Phase L — VS Code / Cursor extension (plan only, for now) ✅ design doc done
- [x] Written design doc — [docs/vscode-extension-design.md](../docs/vscode-extension-design.md). Done ahead of Phase K at the user's explicit request (normally sequenced after it); this is a separate project in its own right (different language/tooling), not a Mazu CLI feature, so implementation is still out of scope until there's a reason to believe people want it. The design pass surfaced a concrete, real prerequisite: Mazu has no machine-readable (`--json`) output today — every command is human-formatted text — so an extension has nothing stable to integrate against yet. That's flagged in the doc as its own future ROADMAP item, not silently assumed away.

### Phase M — Machine-readable output (`--json`)
- [ ] `--json` on the read-oriented commands an integration would actually call: `mazu timeline`, `mazu memory list`, `mazu log`/`mazu log show`, `mazu runs`, `mazu models`. Serializes the same `dict`/`list[dict]` the Python layer underneath `cli.py` already produces (`MemoryStore`, `CheckpointManager`, `ActionLogStore`, `RunStore` methods never returned pre-formatted strings — only `cli.py`'s `click.echo()` calls do) — this is a serialization change, not a new data layer.
- [ ] Structured (not just `--stat`) diff data for `mazu checkpoint diff`/`compare`/`preview_rollback` — currently these wrap `git diff --stat` as text; a real `--json` consumer needs per-file `{path, status, additions, deletions}`, which means parsing `git diff --numstat`/`--name-status` instead, a genuinely separate piece of work from adding `--json` to the already-structured commands above.
- [ ] Version the output shape the same way the CLI itself is versioned (`mazu --version`), so a consumer (e.g. a future VS Code extension, see Phase L) can declare a minimum compatible Mazu version and fail loudly on a mismatch instead of silently misparsing.

This exists because Phase L's design pass found a concrete blocker, not because JSON output is valuable in the abstract: today, anything wanting structured data out of Mazu has to either parse human-formatted terminal text (fragile — nothing currently protects that text from changing) or read the SQLite files directly (couples the caller to an internal schema with no stability promise). Not required before sharing Mazu as a CLI tool; required before anyone can build a serious editor integration on top of it.

### Phase N — Branching checkpoints ✅ done
- [x] `mazu run --from-checkpoint <id> --branch <name> "<task>"` — forks a new, divergent line of execution from any existing checkpoint (git branch + memory.db + skills restored onto it, new run/session id) and runs a task on it, without touching the origin branch's later history. Built on the existing `branch_from()`/`restore()` primitives, not a new mechanism.
- [x] `mazu checkpoint compare-branches <run_id_a> <run_id_b>` — status/steps/stop-reason/memories-saved/estimated-cost side by side for two runs, plus a real diff between their final checkpointed states and their common ancestor checkpoint if one exists.
- [x] Checkpoint index entries gained `branch` and `parent_checkpoint_id` (both additive; older entries simply lack them) so `timeline_entries()`, `prune()`, and no-argument `mazu rollback` are all branch-correct instead of assuming one linear chain — a real, demonstrable bug (wrong diffs, wrong prune targets, rollback targeting the wrong branch) that only shows up once branches coexist, fixed before it could happen in practice rather than after.

### Phase O — Cost-efficiency hardening + crash-safe writes ✅ done
- [x] Multi-breakpoint Anthropic prompt caching — tool schema list and last conversation message now get `cache_control` breakpoints alongside the system prompt (previously system-only), plus `run_forced_tool` (memory extraction/compaction) gained caching it was entirely missing. Live-verified: after step 1 of a real multi-step `mazu run`, each further step showed only ~2 fresh input tokens against 3,900+ read from cache.
- [x] Pricing table gaps closed: `openai:gpt-4o`, `openai:gpt-4o-mini`, `gemini:gemini-2.0-flash` added, so `--max-cost`/`mazu usage`/`mazu models` are no longer silently blind to them.
- [x] `mazu council --max-cost` — a thread-safe shared spend cap across parallel council members (council mode previously had zero cost guardrail, the one place the "minimum cost" goal was actively unmet). Always-on cost-awareness line shown regardless of whether the flag is used.
- [x] Crash-safe (atomic) `write_file`/`edit_file` — write-to-temp-then-`os.replace` so a process/machine crash mid-write can never leave a corrupted file at the real path.
- [x] A real bug found via live testing (not a hypothetical): council members that answered without ever using a tool skipped cost-tracking entirely, silently disabling `--max-cost` for the common no-tool-use case. Fixed and covered by a regression test.

## Directional (not committed — needs a second audience first)

- **Team/shared-memory mode** (shared project memory export/import, approved-memory workflow) — premature for a single-user tool
- **Plugin/tool system** (custom tools folder, manifests, permissions) — premature without other contributors wanting to extend it
- **Usage metrics dashboard** (memory hit rate, time-saved estimates) — needs real usage data to mean anything; building the dashboard before there's data to show is backwards
- **Launch/marketing** (comparison pages, demo video, HN/Reddit/X posts, blog post) — entirely the user's own call, not a code task, and not worth planning until Phases A–J are actually done

## Versioning

No fixed version-to-feature mapping is promised here — each meaningful feature batch gets its own version bump (see git log / PyPI history for what actually shipped when). This file describes direction, not a release schedule.
