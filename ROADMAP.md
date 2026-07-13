# Mazu Roadmap

**Pitch:** Mazu is a local memory and checkpoint layer for coding agents — not a Cursor/Claude Code replacement. It exists to make an agent's history (what it decided, what it changed, what it remembers) visible, reversible, and trustworthy.

This file is split into two parts on purpose:
- **Committed** — what we're actually building next, roughly in order. Each item becomes a real feature with tests before we call it done.
- **Directional (not committed)** — ideas worth keeping on record, revisited once there's a real reason to (more users, a concrete pain point, more time). Not a promise, not a deadline.

## Committed

### Done
- M1–M4: bare tool loop, persistent memory (project + global), checkpoint/rollback, supervised autonomy
- Multi-provider (Anthropic, OpenAI, DeepSeek, Gemini), council mode
- Real-time streaming (`mazu chat`), context compaction (`mazu run`)
- `mazu doctor`, `mazu memory consolidate`, `mazu usage`, `mazu --version`
- Optional semantic memory search (BM25 + embeddings blend)
- Checkpoint UX: `mazu timeline`, `mazu checkpoint show <id>`, `mazu checkpoint diff <id>` — along the way, fixed a real bug where `preview_rollback`/`mazu rollback` showed an empty diff for the most recent checkpoint if there were uncommitted working-tree changes (comparing two commit refs when one side needed to be the live working tree instead), and `git diff` never listing untracked new files at all
- Test suite (210+ tests) + GitHub Actions CI across Python 3.11–3.13 on Linux/Windows/macOS

### Next: Memory UX
The other core differentiator — right now memory is invisible until it's already been injected into context.
- `mazu memory why "<query>"` — shows which memories would be retrieved for a query and why (BM25 score, semantic score if enabled)
- `mazu memory pin <id>` / `unpin <id>` — the `pinned` column already exists in the schema; there's just no command to set it
- `mazu memory stats` — counts by category/source, oldest/newest, superseded count

### Then: Agent Action Log
- Persistent log of every tool call (tool name, input, output summary, changed files, exit code) per session
- `mazu log` / `mazu log show <session_id>`

### Then: Safer Execution
- Shell command allowlist mode (opt-in, alongside the existing denylist)
- `--dry-run` for `mazu run`
- Clearer explanation text when a command is blocked or flagged risky

## Directional (not committed — revisit later, not promised)

- Branch-from-checkpoint (`mazu branch-from <id> <branch-name>`) — create a git branch from a checkpoint without rolling back
- Better autonomous runs: run ids, resume/pause, structured run reports
- Provider layer: model capability table (streaming/tools/context window/cost), `mazu config` for persistent settings, local model support (Ollama/LM Studio)
- Install/onboarding: `mazu doctor --fix`, a `mazu setup` first-run wizard
- `mazu ui` — terminal UI (timeline, memory browser, diff viewer) if the CLI alone stops being enough
- VS Code / Cursor extension — a separate project in its own right, not a Mazu CLI feature; only worth starting once the CLI experience above is solid
- Team/shared-memory mode, plugin/tool system, usage metrics dashboard — all premature for a single-user tool; revisit if/when there's a real second audience
- Launch/marketing (comparison pages, demo video, HN/Reddit posts) — the user's own call, not a code task

## Versioning

No fixed version-to-feature mapping is promised here — each meaningful feature batch gets its own version bump (see git log / PyPI history for what actually shipped when). This file describes direction, not a release schedule.
