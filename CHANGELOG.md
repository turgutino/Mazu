# Changelog

All notable changes to Mazu are documented here, newest first. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); Mazu doesn't yet promise strict semver (see [ROADMAP.md](ROADMAP.md) for what "1.0" would mean) — treat version bumps as "a meaningful batch of work shipped," not a compatibility contract.

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
