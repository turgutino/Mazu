# Mazu for VS Code / Cursor — design document

**Status: design only, not started.** This is Phase L of [ROADMAP.md](../ROADMAP.md), deliberately scoped as a design document, not an implementation task, until there's a real signal that people want it (see [Why this is a separate project](#why-this-is-a-separate-project)). Nothing in this document is committed; it exists so that *if* work starts, it starts from an informed plan instead of a blank page — and so a specific, concrete prerequisite gap in the Mazu CLI itself (documented below) doesn't get discovered halfway through implementation.

## What Mazu currently is

Mazu is a Python CLI (`mazu.cli:main`, a `click.Group`). Every command is a fresh, short-lived process: there is no daemon, no server, no socket/RPC layer, nothing long-running. Each command opens its own SQLite connections under `.mazu/` (project-scoped) or `~/.mazu/` (cross-project — memory and usage), does its work, and exits. See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full module map; this document only covers what's relevant to an editor extension.

## Why this is a separate project

Not a Mazu CLI feature — a VS Code/Cursor extension is a different language (TypeScript), a different packaging/distribution mechanism (the VS Code Marketplace, `vsce`), a different runtime (Node.js inside the editor's extension host), and a different repository structure (most likely its own repo, `mazu-vscode` or similar, not a subdirectory of this Python package). Bundling it into this repo would couple two release cadences and two toolchains that have no reason to be coupled. If this ever gets built, it should be scaffolded as its own project that depends on Mazu (the CLI) being installed, the same way a VS Code Python extension depends on `python` being installed, not the other way around.

## The core prerequisite gap: Mazu has no machine-readable output today

This is the single most important finding of this design pass, and the reason implementation shouldn't start without addressing it first.

**Every command's output is plain, human-formatted text via `click.echo(f"...")`.** There is no `--json` flag, no `--format` option, anywhere in `mazu/cli.py`. `mazu memory list`, `mazu timeline`, `mazu log show`, `mazu runs`, `mazu usage`, `mazu models` — all of it is strings meant for a terminal, padded and aligned for human eyes (`f"{key:<32} {stream:<7} ..."` in `mazu models`, for example). An extension has two ways to get data out of Mazu today, and both are worse than they should be:

1. **Parse the CLI's text output.** Fragile by construction — a wording change in `cli.py` (which nothing currently protects against, since nothing consumes that output programmatically) silently breaks the extension. Not viable for anything beyond a trivial "is this thing installed" check.
2. **Read the SQLite files / `checkpoints/index.json` directly**, bypassing the CLI entirely. More stable than #1, but couples the extension to Mazu's internal schema (table names, column names — see the table below), which isn't a public contract today and could change between Mazu versions without warning. Also awkward from Node.js (needs a SQLite driver, and has to reimplement business logic like "what counts as an active memory" — `superseded_by IS NULL` — that currently lives in Python, not in the schema itself).

Neither is good enough for a real extension. **The actual prerequisite work, before any extension code gets written, is adding a machine-readable read layer to Mazu itself.**

### Recommended shape for that read layer

Not literally speccing this now (that's its own design pass, in this repo, as a normal ROADMAP item if it's ever prioritized) — but the shape that fits best given what already exists:

- Every read-oriented store method already returns `list[dict]` / `dict` / `sqlite3.Row`, not pre-formatted strings — `MemoryStore`, `CheckpointManager` (`inspect_conversation`, `inspect_memory`, `show_entry`, `timeline_entries`, `list_checkpoints`), `ActionLogStore`, `RunStore`, `UsageStore` all already have a clean data layer underneath the CLI's text formatting. `cli.py` is the *only* layer that stringifies them.
- The two exceptions: `CheckpointManager.diff_against_current()`/`compare()`/`preview_rollback()` return raw `git diff --stat` text, not structured per-file data (added lines/removed lines/status). A real diff view in an extension needs this restructured (`git diff --numstat`/`--name-status`, parsed into a list of `{path, status, additions, deletions}`) — this is real, non-trivial work, not just "add `--json`."
- Cheapest viable option: add `--json` to the specific read commands an extension would actually call (`memory list`, `checkpoint list/show`, `timeline`, `log`/`log show`, `runs`, `usage`, `models`), serializing the same dicts the Python layer already produces. Keeps one code path (CLI) as the only integration surface, versioned the same way the CLI itself is versioned (`mazu --version`), so the extension can declare a minimum supported Mazu version and fail loudly on mismatch instead of silently misparsing.
- More involved option: a small internal-only export surface (e.g. `mazu _internal export <resource> --json`, underscore-prefixed to signal "not a public interactive command, no stability promise beyond what the extension itself pins to") that wraps the store classes directly, decoupled from the human-facing command output evolving independently. Worth it only if the human-facing text output and the machine-readable shape genuinely need to diverge; likely overkill for a v1.

Either way: **JSON over stdout from a subprocess, not a socket/HTTP server.** Introducing a long-running daemon is a much bigger step (process lifecycle management, port/socket conflicts, auth if it's ever a shared machine) for a feature that's fundamentally "run a command, get structured data back, render it" — every case below fits a request/response subprocess call.

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

- Actual implementation — this document, not code, is the Phase L deliverable.
- The prerequisite read-layer work in the Python CLI itself — flagged here as a real, concrete blocker, but scoped as its own future ROADMAP item in *this* repo if it's ever prioritized, not part of the extension project.
- An in-editor chat/run UI (see "What the extension would actually show" above) — deliberately excluded from v1 scope, revisit only if the read-only panels prove valuable first.
- JetBrains/other-editor support — VS Code/Cursor only, since Cursor's popularity is exactly the reason this phase exists on the roadmap at all.

## Open questions (not resolved here — for whoever picks this up)

- Does the extension need to *write* `.mazu/` config (e.g. a settings UI for `mazu config set`), or is CLI-only configuration acceptable for v1? (Leaning: CLI-only for v1 — one less UI surface to build and keep in sync.)
- Should the extension auto-detect a Mazu project (`.mazu/` present) and stay dormant otherwise, or always show its panels with an empty/setup state? (Leaning: dormant — matches how most language-specific VS Code extensions activate only when relevant files are present.)
- Live run status (panel 3) requires knowing a `mazu run` is in progress *right now* — from a separate terminal, there's no push notification, only polling `mazu runs`/`RunStore` for `status = 'running'`. Is polling (e.g. every few seconds while a run is detected active) acceptable, or does this actually justify the daemon/socket approach this document otherwise argues against? (Leaning: polling is fine at this scale — a running agent task is on the order of seconds to minutes, not something needing sub-second push updates.)
