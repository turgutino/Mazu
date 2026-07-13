# Mazu

**A memory-augmented, checkpointable coding agent CLI.** Open source, runs entirely on your own machine, works with Anthropic, OpenAI, DeepSeek, or Gemini.

Most coding agents forget everything the moment the session ends. Mazu doesn't. It keeps a real, queryable memory of your project — decisions, conventions, mistakes — that persists across sessions and gets surfaced automatically. And because every autonomous step is checkpointed (code + memory + conversation, together), you can let it run longer and less-supervised without losing the ability to undo it.

```
  ███╗   ███╗
  ████╗ ████║   A Z U
  ██╔████╔██║   ══════════════════════════════
  ██║╚██╔╝██║   memory-augmented coding agent
  ██║ ╚═╝ ██║   persistent memory · checkpoints · skills · multi-model
  ╚═╝     ╚═╝
```

## What makes Mazu different

Compared to typical coding-agent CLIs, which reset context every session and rely on a static instructions file:

1. **Persistent structured memory.** Decisions, conventions, and mistakes made on a project are written to a local SQLite database, ranked by local BM25 (zero API cost — no embedding calls) against your current task, and automatically injected back into context. You never have to re-explain "we use Postgres, not SQLite" in every new session. Optional semantic search (see below) recovers matches BM25's keyword-only ranking would miss entirely.
2. **A separate, global memory for *you*, not the project.** Personal facts — your name, preferred language, experience level, working style — live in `~/.mazu/global_memory.db` and follow you into every project, instead of being repeated (or lost) per-repo.
3. **Checkpointable autonomy.** Every step of an autonomous run snapshots code (via git), the memory database, and the live conversation together. Roll back any one of them and you roll back all three, consistently — "undo" for an agent's actions, not just its files.
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
| `--max-cost USD` | Stop once estimated spend (from a built-in pricing table) reaches this amount |
| `--keep-checkpoints N` | Prune on-disk checkpoint data beyond the N most recent (default 50) |
| `--model provider:model` | Override the model for this run |

By default, file writes/edits proceed unattended in `run` mode (checkpoints make them recoverable), but shell commands still ask for confirmation unless `--allow-shell` is passed. Regardless of that flag, a hardcoded denylist always blocks a short list of genuinely dangerous commands: force-pushing, `sudo`, touching `~/.ssh`, disk-format commands, and `rm -rf /`-style wipes. Checkpoints undo file damage — they can't undo an irreversible external action like a force-push or a sent network request, so that backstop stays even with `--allow-shell`.

`mazu run` refuses to start on a dirty working tree, so the first checkpoint is always a clean baseline. If it's interrupted with **Ctrl-C**, you're offered `[c]ontinue`, `[r]ollback <id>`, or `[q]uit` before anything is lost.

### Cost & usage tracking

Every `mazu chat`, `mazu run`, and `mazu council` call is logged (provider, model, tokens, estimated cost) to a small local store at `~/.mazu/usage.db` — global across every project, since spend is tied to your API keys, not any one codebase.

```bash
mazu usage                  # total estimated spend, all time, broken down by model
mazu usage --since-days 7   # only the last 7 days
```

Like `--max-cost`, this is an *estimate* from a built-in, occasionally-stale pricing table (see [`mazu/llm/pricing.py`](mazu/llm/pricing.py)) — treat it as a helpful approximation, not a substitute for your provider's own billing dashboard.

### Memory

```bash
mazu memory list                    # everything remembered about this project
mazu memory list --category mistake # filter by category
mazu memory list --global           # the cross-project store (facts about you, not the code)
mazu memory forget <id>             # delete a memory by id
mazu memory forget <id> --global
mazu memory consolidate --dry-run   # preview near-duplicate memories that would be merged
mazu memory consolidate             # merge them for real (nothing is deleted — see below)
```

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

### Checkpoints & rollback

```bash
mazu checkpoint                     # manually snapshot code + memory (outside a chat session)
mazu checkpoint list                # list all checkpoints for this project
mazu checkpoint prune --keep 20     # drop old on-disk snapshot copies (git history is untouched)
mazu rollback                       # restore to the most recent checkpoint
mazu rollback cp_000003             # restore to a specific one
```

A checkpoint bundles a git commit, a consistent copy of the memory database (taken via SQLite's online backup API, safe even mid-write), the skill library, and the conversation transcript. Restoring one restores all of them together, so code, what the agent remembers, and what it was talking about never drift out of sync with each other.

### Council mode

```bash
mazu council "should we migrate this service to async I/O, and how risky is it?"
mazu council "..." --models anthropic:claude-sonnet-5,openai:gpt-5,deepseek:deepseek-chat --lead anthropic:claude-opus-4-8
```

Asks each model independently and in parallel (they don't see each other's answers), then has the lead model compare and synthesize a single recommendation. Council members get **read-only** tools only (`read_file`, `list_dir`, `glob_files`, `recall`, `list_skills`) — they can inspect your project to give an informed answer, but can't write, edit, or run anything, so asking several models at once never risks them clobbering each other's changes. This is opt-in and costs one API call per model plus one for the lead — not something you'd want as the default flow for routine tasks.

### Diagnosing setup problems

```bash
mazu doctor          # checks Python/git, which provider keys are set, project readiness
mazu doctor --live   # also makes one minimal real API call per configured key to confirm it actually works
```

Useful when `mazu chat`/`mazu run` fails and it's not obvious why — `mazu doctor --live` will tell you, for example, if a key is set but has been revoked (a plain "not set" check can't catch that, only an actual API call can).

## Model naming

Models are named `provider:model` — e.g. `anthropic:claude-sonnet-5`, `openai:gpt-5`, `deepseek:deepseek-chat`, `deepseek:deepseek-reasoner`, `gemini:gemini-2.0-flash`. A bare name with no prefix (`MAZU_MODEL=claude-opus-4-8`) is assumed to be Anthropic.

Resolution order when you don't pass `--model`:
1. `MAZU_MODEL` environment variable, if set.
2. Auto-detected from whichever provider's API key is actually present in your environment.
3. A hardcoded Anthropic fallback (which just surfaces a clear "set ANTHROPIC_API_KEY" message if you truly have no key configured).

A DeepSeek-only or OpenAI-only setup works with zero extra flags — Anthropic is only a tie-breaker if more than one key happens to be set, never a hard requirement.

## How it fits together

```
mazu/
├── cli.py             Click entry point — chat / run / council / memory / skills / checkpoint / rollback
├── agent/
│   ├── loop.py         interactive chat REPL
│   ├── autonomous.py   unattended multi-step runner with circuit breaker + cost limit
│   ├── council.py      parallel multi-model advisory round + lead synthesis
│   ├── context.py      builds the system prompt from project + global memory + skills
│   └── prompts.py      the system prompt itself
├── llm/
│   ├── client.py        single run_turn()/run_forced_tool() seam every provider call goes through
│   ├── providers/       Anthropic, OpenAI, DeepSeek, Gemini adapters behind a common interface
│   ├── errors.py        normalized error hierarchy (rate limit, auth, transient, context-length)
│   └── pricing.py        rough per-model cost estimates for --max-cost
├── memory/
│   ├── store.py          SQLite-backed memory store (shared by project + global instances)
│   ├── retrieval.py       BM25 ranking (+ optional semantic blending) + context-block rendering
│   ├── embeddings.py      optional semantic search layer (opt-in, see below)
│   └── extraction.py      end-of-session auto-extraction prompt
├── checkpoint/
│   └── manager.py         git commit + memory/skills/conversation snapshot, retention/pruning
├── skills/
│   └── manager.py         save/list/run local skill functions
└── tools/                 read_file, write_file, edit_file, list_dir, glob_files, run_shell,
                           remember, recall, save_skill, run_skill, list_skills
```

The project-scoped memory database lives at `.mazu/memory.db` (created by `mazu init`, gitignored by default). The global, cross-project store lives at `~/.mazu/global_memory.db`, outside any project entirely.

## Security notes

- All file tools are sandboxed to the project root — paths (including through symlinks) that resolve outside it are rejected.
- Shell commands go through a shared denylist (destructive/irreversible patterns) in **both** `mazu chat` and `mazu run`, regardless of confirmation settings.
- API keys are only ever read from environment variables or `~/.mazu/config.toml` (outside any repo) — Mazu never writes a key to disk itself, and `.mazu/`, `.env`, and `config.toml` are all gitignored by default.
- `mazu run` refuses to start with uncommitted changes already present, so autonomous edits are always diffable against a clean baseline.
- See [SECURITY.md](SECURITY.md) for the full policy and how to report a vulnerability.

## Status & roadmap

Milestones M1–M4 (bare tool loop, persistent memory, checkpoint/rollback, supervised autonomy) all have a working implementation, plus multi-provider support (Anthropic, OpenAI, DeepSeek, Gemini), council mode, real-time streaming (`mazu chat`), automatic context compaction (`mazu run`), a setup diagnostic (`mazu doctor`), memory deduplication (`mazu memory consolidate`), and optional semantic memory search on top. This has been exercised through live testing against real Anthropic, OpenAI, and DeepSeek API keys — chat/run tool use, memory recall (project and global), skill save/run, checkpoint/rollback (including pruning and skill restoration), memory supersede, and parallel council queries have all been verified working end-to-end. Semantic search specifically was verified with real embedding calls: a memory with zero shared vocabulary with the query it was retrieved by (proving the blend actually recovers what BM25 alone cannot, not just that it returns numbers). A test suite covers the core logic (memory dedup/supersede, BM25 + semantic blending, checkpoint snapshot/restore, provider routing, streaming response parsing, context-compaction correctness, Gemini's message/tool-call conversion) with zero API cost by default, running on every push via GitHub Actions across Python 3.11–3.13 on Linux/Windows/macOS.

**Known gaps, honestly listed:**
- Semantic search's 50/50 BM25/embedding blend weight is a fixed constant, not tuned against real usage data — may need adjusting once there's more real-world signal on how well it actually ranks.
- Checkpoint/rollback is linear (like `git reset --hard`), not a branching tree.
- Streaming is `mazu chat`-only for now — `mazu run` and `mazu council` still return complete responses, not token-by-token (streaming plus mid-stream confirmation prompts, or interleaved parallel council output, are separate design questions). Gemini specifically doesn't stream at all yet even in `mazu chat` (falls back to a complete response) — its chunk-level behavior for function calls needs to be verified against the live API before building real streaming for it.
- Context compaction is `mazu run`-only for now, for the same reason.
- Live testing so far has been on Windows; Mac/Linux should work (nothing OS-specific in the design, and CI now runs the test suite on all three) but hasn't been verified against a real provider outside of CI's mocked tests.
- Gemini's request/response handling is live-verified for authentication and error classification (a real request reached the API and a real 429 was correctly classified), but not yet for a full successful generation — the API key used for testing had zero free-tier quota available. Treat the Gemini provider as implemented-and-reviewed rather than fully live-proven until that's confirmed.
- `mazu memory consolidate` uses a "keep the newest" heuristic when merging duplicates, which isn't always the most complete entry — always check `--dry-run` output before applying.

Contributions and issue reports are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and distribute, commercially or otherwise, as long as the copyright notice (Turgut Sofuyev) is kept in copies of the software.
