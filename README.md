# Mazu

**Mazu is a local memory and checkpoint layer for coding agents.**

```
  ███╗   ███╗
  ████╗ ████║   A Z U
  ██╔████╔██║   ══════════════════════════════
  ██║╚██╔╝██║   memory-augmented coding agent
  ██║ ╚═╝ ██║   persistent memory · checkpoints · skills · multi-model
  ╚═╝     ╚═╝
```

## What is Mazu?

A CLI coding agent (chat with it, or let it run autonomously) that keeps a real, queryable memory of your project across sessions, and checkpoints its code + memory + conversation together at every step so any point can be rolled back to. Open source, runs entirely on your own machine, works with Anthropic, OpenAI, DeepSeek, or Gemini.

## Why not Cursor / Claude Code?

Mazu isn't trying to replace them, and it isn't an IDE. The goal is narrower and more specific: **manage an agent's memory, its steps, and its ability to undo them** — not write more code faster. If you want a polished in-editor experience with a large tool ecosystem, use Cursor or Claude Code. If you've felt the specific pain of an agent forgetting a project decision it already made last week, or of not trusting a multi-step autonomous run because you can't see what it actually did or easily undo it, that's the gap Mazu fills. It's a complement to those tools' workflow, not a competitor to their editor.

## Core idea: memory + checkpoints

1. **Persistent structured memory.** Decisions, conventions, and mistakes made on a project are written to a local SQLite database, ranked by local BM25 (zero API cost — no embedding calls) against your current task, and automatically injected back into context. You never have to re-explain "we use Postgres, not SQLite" in every new session. Optional semantic search (see below) recovers matches BM25's keyword-only ranking would miss entirely.
2. **A separate, global memory for *you*, not the project.** Personal facts — your name, preferred language, experience level, working style — live in `~/.mazu/global_memory.db` and follow you into every project, instead of being repeated (or lost) per-repo.
3. **Checkpointable autonomy.** Every step of an autonomous run snapshots code (via git), the memory database, and the live conversation together. Roll back any one of them and you roll back all three, consistently — "undo" for an agent's actions, not just its files. `mazu timeline` / `checkpoint show` / `checkpoint diff` make that history inspectable, not just a black box you either trust or roll back blind.
4. **A self-growing local skill library.** When the agent solves something reusable, it can save it as a plain Python function. Next time a similar task comes up, it can run the skill directly — skipping the model call entirely.
5. **Provider-agnostic.** Anthropic, OpenAI, DeepSeek, and Gemini are all first-class, behind one thin adapter interface. Mazu auto-detects which one to use from whichever API key is actually set in your environment — no provider is required over another.
6. **Council mode.** For a decision worth a second opinion, ask two or three different models the same question in parallel and have a lead model synthesize a final recommendation — opt-in, since it costs more than a single call.

Everything above (agent loop, tool execution, memory database, skill library, checkpoints) runs locally. The only network traffic is your chosen model's API call, plus one cheap end-of-session call (on the same provider) to extract memories from the transcript. Nothing else leaves your machine, and there is no server run by this project.

## Installation

Requires **Python 3.11+** and **git** (used for checkpoints).

```bash
pip install mazu

# Only needed for openai:* or deepseek:* models (DeepSeek's API is OpenAI-compatible,
# so it reuses the same client library under a different base URL):
pip install "mazu[openai]"

# Only needed for gemini:* models:
pip install "mazu[gemini]"
```

To work on Mazu itself instead, install it from source in editable mode:

```bash
git clone https://github.com/turgutino/Mazu.git
cd Mazu
pip install -e ".[openai]"
```

Then set at least one API key — Mazu picks whichever provider is present automatically, so you only need the one you actually plan to use:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # or store it in ~/.mazu/config.toml
export DEEPSEEK_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

On Windows (PowerShell): `$env:ANTHROPIC_API_KEY = "sk-ant-..."`

Or skip the manual steps above and run the guided wizard instead:

```bash
mazu setup
```

Walks through picking a provider, pasting its key (persisted to `~/.mazu/config.toml`, never printed back), optionally verifying it with one real API call, optionally setting it as your default model, and initializing the current directory — everything above, in one guided pass.

Something not working? `mazu doctor` (and `mazu doctor --live`) diagnoses the common causes, and `mazu doctor --fix` auto-fixes what's safely fixable (a missing `.gitignore` entry, an uninitialized git repo) — see [Diagnosing setup problems](#diagnosing-setup-problems) below.

## Quick start

```bash
cd your-project/
mazu init                 # creates .mazu/ (local memory + checkpoints) and a git repo if needed
mazu chat                 # start talking to the agent
```

```
> we use PostgreSQL for this project, not SQLite. remember that.
[remember] saved (decision): use PostgreSQL, not SQLite
> /checkpoint
[checkpoint] cp_000001 saved (commit 4f2a91c)
> add a health-check endpoint to app.py and run the tests
...
```

Exit and come back later (even days later) — the next session already knows about PostgreSQL, without you repeating it:

```bash
mazu chat
> what database does this project use?
[memory] loaded prior context relevant to this task

PostgreSQL — this was a project decision, not SQLite.
```

## Usage

### Interactive chat

```bash
mazu chat                       # start a session in the current directory
mazu chat --model deepseek:deepseek-chat
```

Inside the `>` prompt, two extra commands are always available:

| Command | Effect |
|---|---|
| `/checkpoint` | Snapshot code, memory, and the live conversation right now |
| `/rollback [id]` | Restore all three to that checkpoint (defaults to the most recent one) |

Any destructive tool call (writing a file, editing a file, running a shell command) asks for confirmation first. Responses stream in as they're generated (token-by-token where the provider supports it), and each response's `[usage]` line includes a running estimated cost for the session where pricing data is available.

`mazu chat --shell-allowlist "git,npm,pytest"` restricts shell commands to ones starting with a listed program name — opt-in, and additive to the hardcoded denylist below (which always applies regardless).

### Autonomous runs

```bash
mazu run "add input validation to utils.py, run the tests, and fix any failures" --max-steps 15
```

The agent keeps working, unattended, across multiple tool-use rounds until it finishes, hits `--max-steps`, or trips a safety limit. It checkpoints automatically along the way (`--checkpoint-every N`, default every round), so anything it does can be rolled back.

Key flags:

| Flag | Purpose |
|---|---|
| `--max-steps N` | Stop after N tool-use rounds (default 15) |
| `--checkpoint-every N` | Snapshot every N rounds (default 1) |
| `--allow-shell` | Skip the confirmation prompt for shell commands (the hardcoded safety denylist below still applies) |
| `--shell-allowlist "git,npm,pytest"` | Only allow shell commands starting with a listed program name (opt-in, additive to the denylist) |
| `--dry-run` | Preview the task — no files written, no commands executed, no checkpoints created (see below) |
| `--max-cost USD` | Stop once estimated spend (from a built-in pricing table) reaches this amount |
| `--keep-checkpoints N` | Prune on-disk checkpoint data beyond the N most recent (default 50) |
| `--model provider:model` | Override the model for this run |

By default, file writes/edits proceed unattended in `run` mode (checkpoints make them recoverable), but shell commands still ask for confirmation unless `--allow-shell` is passed. Regardless of that flag, a hardcoded denylist always blocks a short list of genuinely dangerous commands: force-pushing, `sudo`, touching `~/.ssh`, disk-format commands, and `rm -rf /`-style wipes — and blocked commands now say *why* (e.g. "elevates privileges via sudo"), not just that a rule matched. Checkpoints undo file damage — they can't undo an irreversible external action like a force-push or a sent network request, so that backstop stays even with `--allow-shell`.

`mazu run` refuses to start on a dirty working tree, so the first checkpoint is always a clean baseline. If it's interrupted with **Ctrl-C**, you're offered `[c]ontinue`, `[r]ollback <id>`, or `[q]uit` before anything is lost.

#### Dry runs

```bash
mazu run "refactor the auth module to use the new session API" --dry-run
```

`--dry-run` runs the task loop for real — the model still reasons, plans, and calls tools — but `write_file`/`edit_file`/`run_shell` report what they *would* do (`[dry-run] Would write 340 bytes to auth.py`) instead of touching disk or a shell. Read-only tools (`read_file`, `list_dir`, `glob_files`, `recall`) still run for real, so the model's plan is grounded in the actual project, not a guess. No checkpoints are created (nothing changed to snapshot), and the usual clean-working-tree requirement is waived, since a preview has no risk of corrupting a checkpoint baseline. Everything the plan would have done is still visible afterward via `mazu log show <session_id>`.

#### Run reports & resuming

Every `mazu run` ends with a structured report — not just scrollback you have to read back through:

```
=== Run report ===
Session: 3f2b1a9c-...
Stop reason: max_steps
Steps: 15/15
Files changed: auth.py, session.py, tests/test_auth.py
Checkpoints created: 15
Memories saved: 2
Tool errors: 0
```

`mazu runs` lists recent invocations (id, status, stop reason, step progress) — this is what gives you the id to act on next:

```bash
mazu runs                        # id, status, stop reason, steps, checkpoints -- across every mazu run
mazu run --resume <run_id>       # continue a run that stopped at --max-steps, hit --max-cost, or was interrupted
```

`--resume` picks up from that run's *last checkpoint* — not from scratch — reusing its original task, model, and options exactly (other flags passed alongside `--resume` are ignored, to avoid a confusing "which value actually won" situation). It's for the run that stopped short of finishing, not a substitute for `mazu chat`'s open-ended conversation. A run that never checkpointed (a `--dry-run`, or one that failed before its first `--checkpoint-every` boundary) has nothing to resume from and says so plainly instead of guessing.

### Checkpoints & rollback

```bash
mazu checkpoint                     # manually snapshot code + memory (outside a chat session)
mazu checkpoint list                # flat list of all checkpoints for this project
mazu checkpoint prune --keep 20     # drop old on-disk snapshot copies (git history is untouched)
mazu rollback                       # restore to the most recent checkpoint
mazu rollback cp_000003             # restore to a specific one
```

A checkpoint bundles a git commit, a consistent copy of the memory database (taken via SQLite's online backup API, safe even mid-write), the skill library, and the conversation transcript. Restoring one restores all of them together, so code, what the agent remembers, and what it was talking about never drift out of sync with each other.

To actually *see* a checkpoint's history instead of just an id list:

```bash
mazu timeline                       # every checkpoint: what changed since the previous one, memory/skills snapshot status
mazu checkpoint show cp_000003      # one checkpoint's full detail — commit, conversation length, snapshot status
mazu checkpoint diff cp_000003      # what's different between that checkpoint and right now (read-only, no rollback)
```

`checkpoint diff` calls out newly created files explicitly, in a separate section — `git diff` itself never lists untracked files no matter what it's compared against, so a file the agent created but never `git add`ed would otherwise silently disappear from the diff.

A few more ways to work with checkpoint history without touching your current state:

```bash
mazu checkpoint inspect cp_000003 --memory         # the memories actually captured in that snapshot
mazu checkpoint inspect cp_000003 --conversation   # the conversation transcript captured in that snapshot
mazu checkpoint compare cp_000001 cp_000005        # diff between two checkpoints directly (not either vs. now)
mazu branch-from cp_000003 my-experiment           # new git branch at that commit, current branch untouched
```

`branch-from` is deliberately lighter than `mazu rollback`: it only creates a git branch pointer, it never resets your working tree or restores memory/skills, and your current branch stays checked out. It's for exploring "what if I'd gone a different way from here" without the heavier, stateful implications of a full rollback — switch into it yourself with `git checkout my-experiment` when you're ready.

### Memory

```bash
mazu memory list                    # everything remembered about this project
mazu memory list --category mistake # filter by category
mazu memory list --global           # the cross-project store (facts about you, not the code)
mazu memory forget <id>             # delete a memory by id
mazu memory forget <id> --global
mazu memory consolidate --dry-run   # preview near-duplicate memories that would be merged
mazu memory consolidate             # merge them for real (nothing is deleted — see below)
mazu memory why "<query>"           # what would be retrieved for this query, and why (score, or pinned/mistake floor)
mazu memory pin <id>                # always include this memory in context, regardless of relevance ranking
mazu memory unpin <id>
mazu memory edit <id> --title "..." --body "..."   # correct a memory in place instead of forget + re-add
mazu memory supersede <old-id> <new-id>            # mark old as replaced by new (nothing deleted, just retired)
mazu memory stats                   # counts by category/source, pinned/superseded totals, oldest/newest
```

Every subcommand above takes `--global` to operate on the cross-project store instead of this project's.

`mazu memory why` mirrors the exact selection logic used when building a session's context block, so it's a real answer to "why didn't the agent know X" or "why did it bring up Y" — not a separate approximation. `mazu memory list` also shows how many times each memory has actually been rendered into a session's context (`used Nx, last: ...`), tracked only on real retrieval, not on inspection commands like `why` or `list` themselves.

Memory categories: `decision`, `convention`, `mistake`, `task_outcome`, `fact` (all project-scoped) and `user_preference` (global — your name, language, experience level, working style; injected into every project's context automatically).

Memories are written two ways: explicitly, when the agent calls `remember` (you can just tell it "remember that..."), and automatically, via a cheap end-of-session pass that extracts anything notable you didn't ask it to remember. Both paths de-duplicate against existing memories (exact and fuzzy title matching) so the same fact doesn't pile up across sessions, and an explicit `remember` call can mark an older memory as superseded when something changes.

`mazu memory consolidate` catches near-duplicates that slipped past that automatic dedup (entirely local, no API calls — a word-overlap similarity check, same one `remember` already uses). It keeps the **most recently created** entry in each duplicate group and marks the rest as superseded (nothing is deleted, superseded rows just stop showing up). That "keep the newest" rule is a simple heuristic, not a judgment about which entry is actually more complete or useful — a newer, sparser restatement of a fact can end up kept over an older, more detailed one. **Always run `--dry-run` first** and read what it proposes before applying; if it picked the wrong one, `mazu memory forget <id>` the survivor and keep the original.

#### Semantic search (optional)

BM25 only finds memories that share actual words with your current task — it can't tell that "the project's database is Postgres" and "which relational system backs this application" are the same question. Semantic search recovers that, by comparing meaning (via embeddings) instead of just vocabulary.

It's **off by default** and entirely opt-in, since it adds a real (small) API cost per `remember` call and per retrieval, on top of whatever your main model already costs:

```bash
export MAZU_SEMANTIC_MEMORY=1
export OPENAI_API_KEY=sk-...   # required even if your main model is a different provider
pip install "mazu[openai]"     # if not already installed
```

With that set, new memories get an embedding (OpenAI's `text-embedding-3-small`) stored alongside them at write time, and retrieval blends BM25 with cosine similarity (50/50) instead of ranking on keywords alone. Memories written before you turned this on simply don't have a stored embedding and fall back to their BM25 contribution — nothing breaks, nothing needs to be backfilled. Requires an OpenAI key specifically (for the embedding call) regardless of which provider you use for the main conversation.

### Skills

```bash
mazu skills list                    # saved reusable solutions for this project
mazu skills forget <name>           # delete one
```

When the agent solves a reusable problem, it can save the solution as a plain Python function under `.mazu/skills/<name>/`. The next time a similar task comes up, it can call the skill directly instead of solving it again from scratch through the model — a real cost and latency win for repeated, mechanical work.

### Council mode

```bash
mazu council "should we migrate this service to async I/O, and how risky is it?"
mazu council "..." --models anthropic:claude-sonnet-5,openai:gpt-5,deepseek:deepseek-chat --lead anthropic:claude-opus-4-8
```

Asks each model independently and in parallel (they don't see each other's answers), then has the lead model compare and synthesize a single recommendation. Council members get **read-only** tools only (`read_file`, `list_dir`, `glob_files`, `recall`, `list_skills`) — they can inspect your project to give an informed answer, but can't write, edit, or run anything, so asking several models at once never risks them clobbering each other's changes. This is opt-in and costs one API call per model plus one for the lead — not something you'd want as the default flow for routine tasks.

### Cost & usage tracking

Every `mazu chat`, `mazu run`, and `mazu council` call is logged (provider, model, tokens, estimated cost) to a small local store at `~/.mazu/usage.db` — global across every project, since spend is tied to your API keys, not any one codebase.

```bash
mazu usage                  # total estimated spend, all time, broken down by model
mazu usage --since-days 7   # only the last 7 days
```

Like `--max-cost`, this is an *estimate* from a built-in, occasionally-stale pricing table (see [`mazu/llm/pricing.py`](mazu/llm/pricing.py)) — treat it as a helpful approximation, not a substitute for your provider's own billing dashboard.

### Agent action log

Every tool call `mazu chat`, `mazu run`, and `mazu council` make — what tool, what input, what happened, and what file (if any) it touched — is recorded to a project-scoped store at `.mazu/action_log.db`, so "what exactly did the agent do in this session" is always answerable after the fact, not just while you're watching the terminal.

```bash
mazu log                       # recent sessions: action count, error count, time range
mazu log show <session_id>     # every tool call in that session, in order, with input/output
```

This captures the tool-call layer specifically — blocked (denylisted shell command), declined (you said no to a confirmation prompt), and unknown-tool cases are all recorded too, not just successful calls, so a `mazu log show` after a confusing run tells you exactly where and why something didn't happen the way you expected.

### Diagnosing setup problems

```bash
mazu doctor          # checks Python/git, which provider keys are set, project readiness
mazu doctor --live   # also makes one minimal real API call per configured key to confirm it actually works
mazu doctor --fix    # auto-fixes what's safely fixable: missing .gitignore entry, uninitialized git repo
```

Useful when `mazu chat`/`mazu run` fails and it's not obvious why — `mazu doctor --live` will tell you, for example, if a key is set but has been revoked (a plain "not set" check can't catch that, only an actual API call can).

`--fix` deliberately only touches things that have exactly one correct answer and no downside — it never touches API keys, your Python installation, or installs packages, since those need a value or a decision only you have. Use `mazu setup` for API keys instead.

## Model naming

Models are named `provider:model` — e.g. `anthropic:claude-sonnet-5`, `openai:gpt-5`, `deepseek:deepseek-chat`, `deepseek:deepseek-reasoner`, `gemini:gemini-2.0-flash`. A bare name with no prefix (`MAZU_MODEL=claude-opus-4-8`) is assumed to be Anthropic.

Resolution order when you don't pass `--model`:
1. `MAZU_MODEL` environment variable, if set.
2. `default_model` set via `mazu config set` (see below), if set.
3. Auto-detected from whichever provider's API key is actually present in your environment.
4. A hardcoded Anthropic fallback (which just surfaces a clear "set ANTHROPIC_API_KEY" message if you truly have no key configured).

A DeepSeek-only or OpenAI-only setup works with zero extra flags — Anthropic is only a tie-breaker if more than one key happens to be set, never a hard requirement.

### Persistent settings

```bash
mazu config set default_model deepseek:deepseek-chat   # skip MAZU_MODEL / --model going forward
mazu config set anthropic_api_key sk-ant-...            # alternative to an env var, same effect
mazu config list                                        # secrets are masked, showing only the last 4 characters
mazu config unset default_model
```

Stored in `~/.mazu/config.toml`, outside any repo. **Environment variables always win** — a config-file value is only used if the corresponding env var (`MAZU_MODEL`, `ANTHROPIC_API_KEY`, etc.) isn't already set, exactly like the resolution order above. Known keys: `default_model`, `anthropic_api_key`, `openai_api_key`, `deepseek_api_key`, `gemini_api_key` (plus a legacy `api_key`, kept as an alias for `anthropic_api_key` for anyone with a config file from before per-provider keys existed).

### Model capabilities

```bash
mazu models
```

Lists every model Mazu knows about with real (token-by-token) streaming support, tool-use support, approximate context window, and approximate pricing side by side — useful for picking a model without cross-referencing four different providers' docs. Best-effort and may go stale; see [`mazu/llm/capabilities.py`](mazu/llm/capabilities.py).

## Safety model

- All file tools are sandboxed to the project root — paths (including through symlinks) that resolve outside it are rejected.
- Shell commands go through a shared denylist (destructive/irreversible patterns) in **both** `mazu chat` and `mazu run`, regardless of confirmation settings.
- API keys are only ever read from environment variables or `~/.mazu/config.toml` (outside any repo) — Mazu never writes a key to disk itself, and `.mazu/`, `.env`, and `config.toml` are all gitignored by default.
- `mazu run` refuses to start with uncommitted changes already present, so autonomous edits are always diffable against a clean baseline.
- See [SECURITY.md](SECURITY.md) for the full policy and how to report a vulnerability.

## Architecture

The module map, the provider adapter seam, and how memory/checkpoints/skills fit together are documented separately in [ARCHITECTURE.md](ARCHITECTURE.md), so this README can stay focused on using Mazu rather than building it.

## Status & roadmap

Milestones M1–M4 (bare tool loop, persistent memory, checkpoint/rollback, supervised autonomy) all have a working implementation, plus multi-provider support, council mode, real-time streaming (`mazu chat`), context compaction (`mazu run`), a setup diagnostic, memory deduplication, checkpoint history inspection (`mazu timeline`/`checkpoint show`/`checkpoint diff`), and optional semantic memory search. This has been exercised through live testing against real Anthropic, OpenAI, DeepSeek, and Gemini API keys, plus a test suite (210+ tests, zero API cost by default) running on every push via GitHub Actions across Python 3.11–3.13 on Linux/Windows/macOS.

**Known gaps, honestly listed:**
- Semantic search's 50/50 BM25/embedding blend weight is a fixed constant, not tuned against real usage data.
- Checkpoint/rollback is linear (like `git reset --hard`), not a branching tree yet.
- Streaming is `mazu chat`-only for now; Gemini doesn't stream at all yet (falls back to a complete response, correct but not incremental — see `mazu models`). Deliberately deferred: the `google-genai` SDK's docs and source don't say whether function-call parts arrive fragmented, whole, or only in the final chunk during a stream, and guessing wrong risks silently mishandling tool calls. Needs a live Gemini key with real quota to verify before implementing.
- Live testing so far has been on Windows; CI runs the test suite on Linux/macOS too, but a real provider hasn't been exercised live on those platforms yet.
- Gemini is live-verified for authentication and error classification, but not yet for a full successful generation (the test key had zero free-tier quota).
- `mazu memory consolidate` uses a "keep the newest" heuristic, which isn't always the most complete entry — check `--dry-run` output before applying.

The full build sequence — what's next, in what order, and what's deliberately not planned yet — is in [ROADMAP.md](ROADMAP.md).

Contributions and issue reports are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and distribute, commercially or otherwise, as long as the copyright notice (Turgut Sofuyev) is kept in copies of the software.
