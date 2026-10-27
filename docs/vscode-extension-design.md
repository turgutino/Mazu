# Mazu for VS Code / Cursor — design document

**Status: design only, not started, and not currently planned** (see [ROADMAP.md](../ROADMAP.md), "Directional"). It is written down so that *if* work starts, it starts from an informed plan, and so the remaining prerequisite in the Mazu CLI itself (structured diffs, below) is known up front. Nothing in this document is committed.

## What Mazu currently is

Mazu is a Python CLI (`mazu.cli:main`, a `click.Group`). Every command is a fresh, short-lived process: there is no daemon, no server, no socket/RPC layer, nothing long-running. Each command opens its own SQLite connections under `.mazu/` (project-scoped) or `~/.mazu/` (cross-project — memory and usage), does its work, and exits. See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full module map; this document only covers what's relevant to an editor extension.

## Why this is a separate project

Not a Mazu CLI feature — a VS Code/Cursor extension is a different language (TypeScript), a different packaging/distribution mechanism (the VS Code Marketplace, `vsce`), a different runtime (Node.js inside the editor's extension host), and a different repository structure (most likely its own repo, `mazu-vscode` or similar, not a subdirectory of this Python package). Bundling it into this repo would couple two release cadences and two toolchains that have no reason to be coupled. If this ever gets built, it should be scaffolded as its own project that depends on Mazu (the CLI) being installed, the same way a VS Code Python extension depends on `python` being installed, not the other way around.

## The integration surface: JSON over a subprocess

Originally this was the main blocker: every command printed human-formatted text only, so an extension could either parse fragile terminal output or read the SQLite files directly and couple itself to an unversioned schema. That gap is **mostly closed**:

- `mazu timeline`, `mazu memory list`, `mazu log` / `mazu log show`, `mazu runs`, and `mazu models` accept `--json` and emit one versioned envelope, `{"schema_version": 1, "mazu_version": "...", "data": ...}`, built by a single helper (`mazu/output.py`). An extension can declare a minimum Mazu version and fail loudly on a mismatch instead of silently misparsing.
- `mazu chat --list --json` lists past chat sessions the same way.
- The data layer underneath was already structured: `MemoryStore`, `CheckpointManager`, `ActionLogStore`, `RunStore`, and `UsageStore` return dicts and rows; only `cli.py` turns them into text.

**Still missing:**

- **Structured diffs.** `checkpoint diff`, `compare`, and the rollback preview wrap `git diff --stat` text. A real diff view needs per-file `{path, status, additions, deletions}`, which means parsing `git diff --numstat` and `--name-status`. This is real work, not just a flag, and it blocks the checkpoint-diff panel below.
- `mazu usage` and `mazu checkpoint list/show` do not have `--json` yet.

The integration should stay **JSON over stdout from a subprocess, not a socket or HTTP server.** A long-running daemon brings process lifecycle, port and socket conflicts, and authentication concerns for a feature that is fundamentally "run a command, get structured data, render it."

## What the extension would actually show

Three read-only panels, all backed by the read layer above, in priority order:

1. **Checkpoint timeline** — the editor-native version of `mazu timeline`: a list/tree view of checkpoints (id, timestamp, trigger, files changed), clicking one opens a diff view (needs the structured-diff prerequisite above), with actions to roll back or branch-from directly from the UI instead of switching to a terminal.
2. **Memory browser** — the editor-native version of `mazu memory list` + `mazu memory why`: browse project memory by category, see pin state and retrieval stats, edit/pin/forget from the UI. `mazu memory why "<query>"`'s scoring output (Phase C) maps well to a "why was this shown" inline explanation.
3. **Action log / run status** — the editor-native version of `mazu log show` + `mazu runs`: while a `mazu run` is executing (started either from the extension or from a terminal Mazu happens to be running in), show live tool-call activity and the end-of-run report inline instead of scrolling terminal output.

Explicitly **not** in a v1: driving `mazu chat`/`mazu run` itself from inside the extension (i.e., an in-editor chat panel that talks to the LLM). That's a much larger surface (streaming token-by-token into a webview, matching the terminal's confirmation-prompt safety flow in a GUI context, multi-provider model selection UI) and duplicates work Cursor/Claude Code/Copilot already do well. Mazu's actual differentiation — persistent memory and checkpoints — is exactly the read-only, inspect-and-recover surface described above, not a chat UI competing head-on with tools built for that specifically.

## Tech stack (if built)

- **TypeScript**, VS Code Extension API (`vscode` package) — the only realistic choice for a VS Code/Cursor extension (Cursor is a VS Code fork and uses the same extension API).
- **TreeView API** for the checkpoint timeline and memory browser (VS Code's native tree UI — `TreeDataProvider`), not a custom webview, for the list views — cheaper to build, more consistent with the rest of the editor, keyboard-navigable for free.
- **Webview** only for the diff view (needs custom rendering beyond what `TreeView` offers) and possibly the live run-status panel (needs a scrolling log-style view).
- Extension spawns `mazu` as a child process (Node's `child_process.execFile`) per data request — no persistent connection, matching Mazu's own "every command is a fresh process" model on the other side.
- Packaging: standard `vsce package`/Marketplace publish flow, own `package.json`/`CHANGELOG.md`, own versioning (semver, independent of Mazu's own version number, just declaring a minimum compatible Mazu CLI version).

## Safety model carries over unchanged

The extension must not weaken any safety property the CLI already has: destructive actions (rollback, forget, branch-from with an existing name) still go through the same confirmation the CLI would show, just as a VS Code `showWarningMessage`/modal instead of a terminal prompt. The extension is a *view* into Mazu's existing safety model (denylist, dry-run, confirmation-gated destructive tools), not a second implementation of it — every mutating action the extension offers should shell out to the real `mazu` command (`mazu rollback`, `mazu memory forget`, etc.) rather than reimplementing the mutation logic against the SQLite files directly, so there's exactly one place (the Python CLI) that enforces the safety rules.

## Explicitly out of scope for this design pass

- Actual implementation.
- The structured-diff work in the Python CLI itself, which would be its own item in *this* repo if prioritized, not part of the extension project.
- An in-editor chat/run UI (see "What the extension would actually show" above) — deliberately excluded from v1 scope, revisit only if the read-only panels prove valuable first.
- JetBrains/other-editor support — VS Code/Cursor only, since Cursor's popularity is exactly the reason this phase exists on the roadmap at all.

## Open questions (not resolved here — for whoever picks this up)

- Does the extension need to *write* `.mazu/` config (e.g. a settings UI for `mazu config set`), or is CLI-only configuration acceptable for v1? (Leaning: CLI-only for v1 — one less UI surface to build and keep in sync.)
- Should the extension auto-detect a Mazu project (`.mazu/` present) and stay dormant otherwise, or always show its panels with an empty/setup state? (Leaning: dormant — matches how most language-specific VS Code extensions activate only when relevant files are present.)
- Live run status (panel 3) requires knowing a `mazu run` is in progress *right now* — from a separate terminal, there's no push notification, only polling `mazu runs`/`RunStore` for `status = 'running'`. Is polling (e.g. every few seconds while a run is detected active) acceptable, or does this actually justify the daemon/socket approach this document otherwise argues against? (Leaning: polling is fine at this scale — a running agent task is on the order of seconds to minutes, not something needing sub-second push updates.)
