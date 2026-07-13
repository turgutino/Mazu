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

### Phase B — Checkpoint UX
- [x] `mazu timeline`
- [x] `mazu checkpoint show <id>`
- [x] `mazu checkpoint diff <id>`
- [ ] `mazu checkpoint inspect <id> --memory` / `--conversation` — print a snapshot's captured memory rows / conversation transcript directly, without restoring
- [ ] `mazu checkpoint compare <id1> <id2>` — diff between two checkpoints (not just checkpoint-vs-current)
- [ ] `mazu branch-from <id> <branch-name>` — create a git branch from a checkpoint's commit without rolling back the current one

### Phase C — Memory UX
- [ ] `mazu memory why "<query>"` — shows which memories would be retrieved and why (BM25 score, semantic score if enabled)
- [ ] `mazu memory pin <id>` / `unpin <id>` — the `pinned` column already exists; just needs a command
- [ ] `mazu memory stats` — counts by category/source, oldest/newest, superseded count
- [ ] `mazu memory edit <id>` — edit an existing memory's title/body in place
- [ ] `mazu memory supersede <old_id> <new_id>` — CLI wrapper around the `supersede()` method `remember` already uses internally
- [ ] Track `last_used_at` / `retrieval_count` per memory (new columns + update-on-retrieval), shown in `mazu memory list`

### Phase D — Agent Action Log
- [ ] Persistent log of every tool call: time, tool name, input, output summary, changed files, exit code
- [ ] `mazu log` — list recent sessions/runs
- [ ] `mazu log show <session_id>` — full action log for one session

### Phase E — Safer Execution
- [ ] `mazu run "..." --dry-run` — show what would happen without writing/running anything
- [ ] Shell command allowlist mode (opt-in, alongside the existing denylist)
- [ ] Clearer explanation text when a command is blocked or flagged risky

### Phase F — Better Autonomous Runs
- [ ] Run id system (already have session_id; formalize as a first-class concept in `mazu run`)
- [ ] `mazu run resume <run_id>`
- [ ] Structured end-of-run report: files changed, memories saved, checkpoints created, errors hit

### Phase G — Provider Layer
- [ ] Model capability table (streaming support, tool support, context window, approximate cost) surfaced somewhere inspectable (`mazu doctor` or a new command)
- [ ] `mazu config set / list` for persistent settings (default model, etc.) instead of only env vars
- [ ] Real streaming for Gemini once its chunk-level function-call behavior is verified against the live API (see README's known gaps)

### Phase H — Install & Onboarding
- [ ] `mazu doctor --fix` — auto-fix what it can (missing `.gitignore` entry, offer to run `mazu init`)
- [ ] `mazu setup` — first-run wizard (pick a provider, paste a key, verify it works)

### Phase I — Documentation overhaul
- [ ] README restructured: what is Mazu / why not Cursor / core idea / quickstart / checkpoint demo / memory demo / safety model / architecture (linking out to `ARCHITECTURE.md`) / roadmap
- [ ] `examples/` — at least one worked-through example project
- [ ] `docs/` if the above grows past what fits comfortably in a few root-level files

### Phase J — Demo
- [ ] A concrete, scripted demo scenario (agent makes a mistake → `mazu timeline` finds the bad step → `mazu checkpoint diff` shows the damage → recover via rollback or `branch-from` → memory means it isn't repeated) — as a `docs/demo.md` walkthrough first; a recorded video/GIF is the user's own call, not a code task

### Phase K — Terminal UI (`mazu ui`)
- [ ] Only after Phases B–D are solid — a TUI over checkpoint timeline / memory browser / action log, likely via `textual` or `rich`

### Phase L — VS Code / Cursor extension (plan only, for now)
- [ ] Written design doc once Phase K is done — this is a separate project in its own right (different language/tooling), not a Mazu CLI feature; actual implementation is out of scope until there's a reason to believe people want it

## Directional (not committed — needs a second audience first)

- **Team/shared-memory mode** (shared project memory export/import, approved-memory workflow) — premature for a single-user tool
- **Plugin/tool system** (custom tools folder, manifests, permissions) — premature without other contributors wanting to extend it
- **Usage metrics dashboard** (memory hit rate, time-saved estimates) — needs real usage data to mean anything; building the dashboard before there's data to show is backwards
- **Launch/marketing** (comparison pages, demo video, HN/Reddit/X posts, blog post) — entirely the user's own call, not a code task, and not worth planning until Phases A–J are actually done

## Versioning

No fixed version-to-feature mapping is promised here — each meaningful feature batch gets its own version bump (see git log / PyPI history for what actually shipped when). This file describes direction, not a release schedule.
