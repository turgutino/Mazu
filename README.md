# Mazu

**A memory-augmented, checkpointable coding agent CLI.** Open source, runs entirely on your own machine, works with Anthropic, OpenAI, or DeepSeek.

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

1. **Persistent structured memory.** Decisions, conventions, and mistakes made on a project are written to a local SQLite database, ranked by local BM25 (zero API cost — no embedding calls) against your current task, and automatically injected back into context. You never have to re-explain "we use Postgres, not SQLite" in every new session.
2. **A separate, global memory for *you*, not the project.** Personal facts — your name, preferred language, experience level, working style — live in `~/.mazu/global_memory.db` and follow you into every project, instead of being repeated (or lost) per-repo.
3. **Checkpointable autonomy.** Every step of an autonomous run snapshots code (via git), the memory database, and the live conversation together. Roll back any one of them and you roll back all three, consistently — "undo" for an agent's actions, not just its files.
4. **A self-growing local skill library.** When the agent solves something reusable, it can save it as a plain Python function. Next time a similar task comes up, it can run the skill directly — skipping the model call entirely.
5. **Provider-agnostic.** Anthropic, OpenAI, and DeepSeek are all first-class, behind one thin adapter interface. Mazu auto-detects which one to use from whichever API key is actually set in your environment — no provider is required over another.
6. **Council mode.** For a decision worth a second opinion, ask two or three different models the same question in parallel and have a lead model synthesize a final recommendation — opt-in, since it costs more than a single call.

Everything above (agent loop, tool execution, memory database, skill library, checkpoints) runs locally. The only network traffic is your chosen model's API call, plus one cheap end-of-session call (on the same provider) to extract memories from the transcript. Nothing else leaves your machine, and there is no server run by this project.

## Installation

Requires **Python 3.11+** and **git** (used for checkpoints).

```bash
pip install mazu

# Only needed for openai:* or deepseek:* models (DeepSeek's API is OpenAI-compatible,
# so it reuses the same client library under a different base URL):
pip install "mazu[openai]"
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

Any destructive tool call (writing a file, editing a file, running a shell command) asks for confirmation first.

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

### Memory

```bash
mazu memory list                    # everything remembered about this project
mazu memory list --category mistake # filter by category
mazu memory list --global           # the cross-project store (facts about you, not the code)
mazu memory forget <id>             # delete a memory by id
mazu memory forget <id> --global
```

Memory categories: `decision`, `convention`, `mistake`, `task_outcome`, `fact` (all project-scoped) and `user_preference` (global — your name, language, experience level, working style; injected into every project's context automatically).

Memories are written two ways: explicitly, when the agent calls `remember` (you can just tell it "remember that..."), and automatically, via a cheap end-of-session pass that extracts anything notable you didn't ask it to remember. Both paths de-duplicate against existing memories (exact and fuzzy title matching) so the same fact doesn't pile up across sessions, and an explicit `remember` call can mark an older memory as superseded when something changes.

### Skills

```bash
mazu skills list                    # saved reusable solutions for this project
mazu skills forget <name>           # delete one
```

When the agent solves a reusable problem, it can save the solution as a plain Python function under `.mazu/skills/<name>/`. The next time a similar task comes up, it can call the skill directly instead of solving it again from scratch through the model — a real cost and latency win for repeated, mechanical work.

### Checkpoints & rollback

```bash
mazu checkpoint                     # manually snapshot code + memory (outside a chat session)
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

## Model naming

Models are named `provider:model` — e.g. `anthropic:claude-sonnet-5`, `openai:gpt-5`, `deepseek:deepseek-chat`, `deepseek:deepseek-reasoner`. A bare name with no prefix (`MAZU_MODEL=claude-opus-4-8`) is assumed to be Anthropic.

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
│   ├── providers/       Anthropic, OpenAI, DeepSeek adapters behind a common interface
│   ├── errors.py        normalized error hierarchy (rate limit, auth, transient, context-length)
│   └── pricing.py        rough per-model cost estimates for --max-cost
├── memory/
│   ├── store.py          SQLite-backed memory store (shared by project + global instances)
│   ├── retrieval.py       BM25 ranking + context-block rendering
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

## Status & roadmap

Milestones M1–M4 (bare tool loop, persistent memory, checkpoint/rollback, supervised autonomy) all have a working implementation, plus multi-provider support and council mode on top. This has been exercised through live testing against real Anthropic, OpenAI, and DeepSeek API keys — chat/run tool use, memory recall (project and global), skill save/run, checkpoint/rollback (including pruning and skill restoration), memory supersede, and parallel council queries have all been verified working end-to-end.

**Known gaps, honestly listed:**
- No automated test suite yet (`pytest`) — verification so far has been live testing plus ad-hoc smoke scripts, not a checked-in regression suite.
- No semantic/embedding-based memory retrieval yet — BM25 is a solid, zero-cost baseline, but pure keyword ranking can miss a relevant memory that's phrased very differently from the current task.
- Checkpoint/rollback is linear (like `git reset --hard`), not a branching tree.
- No context compaction for very long autonomous runs — a run that grows the conversation past a model's context window will surface a clear error rather than crash, but won't automatically summarize and continue.
- Windows-only testing so far; Mac/Linux should work (nothing OS-specific in the design) but hasn't been verified live.
- No `mazu memory consolidate` command yet for manually merging/cleaning up accumulated memories.
- No Google Gemini provider yet — the adapter interface is designed to make this a small addition, not a redesign.

Contributions and issue reports are welcome.

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and distribute, commercially or otherwise, as long as the copyright notice (Turgut Sofuyev) is kept in copies of the software.
