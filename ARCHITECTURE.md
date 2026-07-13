# Architecture

This is the system-design detail split out of [README.md](README.md) so the README can stay focused on *using* Mazu. Read this if you're contributing, debugging, or just curious how the pieces fit.

## Module map

```
mazu/
├── cli.py             Click entry point — chat / run / council / memory / skills / checkpoint / rollback / timeline / usage / doctor
├── diagnostics.py      mazu doctor's checks (Python/git availability, key presence, live auth check)
├── agent/
│   ├── loop.py          interactive chat REPL (streaming, /checkpoint, /rollback)
│   ├── autonomous.py     unattended multi-step runner — circuit breaker, cost limit, context compaction
│   ├── council.py        parallel multi-model advisory round + lead synthesis
│   ├── compaction.py     summarizes the oldest part of a long conversation to stay under the context budget
│   ├── context.py        builds the system prompt from project + global memory + skills
│   ├── session.py        end-of-session auto-extraction (the memory safety net)
│   ├── interaction.py    safe_confirm() — EOF/interrupt-safe confirmation prompts
│   └── prompts.py        the system prompt itself, including Mazu's own identity disclosure
├── llm/
│   ├── client.py          single run_turn() / run_turn_stream() / run_forced_tool() seam every provider call goes through
│   ├── providers/         Anthropic, OpenAI, DeepSeek, Gemini adapters behind a common Provider interface
│   ├── errors.py          normalized error hierarchy (MazuRateLimitError, MazuAuthError, MazuTransientError, MazuContextLengthError)
│   ├── error_mapping.py   maps Anthropic/OpenAI SDK exceptions onto that hierarchy (Gemini has its own classifier — see providers/gemini_provider.py)
│   ├── retry.py           exponential backoff for the two retryable error types only
│   └── pricing.py          rough per-model cost estimates for --max-cost / mazu usage
├── memory/
│   ├── store.py            SQLite-backed memory store (the same class backs both the project and global instances)
│   ├── retrieval.py         BM25 ranking (+ optional semantic blending) + context-block rendering
│   ├── embeddings.py         optional semantic search layer (opt-in via MAZU_SEMANTIC_MEMORY)
│   ├── extraction.py         end-of-session auto-extraction prompt (the cheap-model safety net)
│   ├── consolidate.py         near-duplicate detection/merging (mazu memory consolidate)
│   ├── bm25.py                 pure-Python Okapi BM25, zero dependencies
│   └── schema.sql              the memories/sessions table definitions
├── checkpoint/
│   ├── manager.py           git commit + memory/skills/conversation snapshot; retention/pruning; timeline/show/diff
│   └── store.py              the flat JSON checkpoint index (not SQLite — human-readable/diffable at MVP scale)
├── skills/
│   └── manager.py             save/list/run local skill functions (subprocess, args via stdin)
├── usage/
│   └── store.py                 cross-project spend log (~/.mazu/usage.db)
└── tools/                       read_file, write_file, edit_file, list_dir, glob_files, run_shell,
                                  remember, recall, save_skill, run_skill, list_skills
```

## The provider seam

Every LLM call in Mazu goes through exactly one place: `mazu/llm/client.py`'s `run_turn()` / `run_turn_stream()` / `run_forced_tool()`. Nothing else in the codebase imports `anthropic`, `openai`, or `google.genai` directly. Adding a new provider means:

1. Implement the `Provider` ABC (`mazu/llm/providers/base.py`): `run_turn`, `run_forced_tool`, optionally `run_turn_stream` (falls back to a safe non-streaming default if you don't override it).
2. Convert between Mazu's canonical Anthropic-shaped message/tool format and whatever the provider's SDK expects. This is usually the hardest part — see `gemini_provider.py`'s `_to_gemini_contents` for an example of the kind of structural mismatch to expect (different role names, tool-result blocks needing a lookup instead of carrying everything themselves).
3. Register the provider in `client.py`'s `_PROVIDERS` dict and `_PROVIDER_DEFAULT_MODELS`.

Errors from every provider get normalized into one hierarchy (`mazu/llm/errors.py`) so callers (`agent/loop.py`, `agent/autonomous.py`, `agent/council.py`) only ever need to catch `MazuAPIError` and its subclasses, never provider-specific exception types.

## Memory

Two `MemoryStore` instances exist per session, both backed by the same class, different files:
- **Project-scoped** (`.mazu/memory.db`) — decisions, conventions, mistakes, task outcomes, facts. Gitignored by default.
- **Global** (`~/.mazu/global_memory.db`) — `user_preference` category only: facts about *you*, not the codebase, shared across every project.

Retrieval (`memory/retrieval.py`) ranks candidates with local BM25 (zero API cost) and, if `MAZU_SEMANTIC_MEMORY` is set, blends in cosine similarity against a stored embedding (`memory/embeddings.py`). Pinned memories and the 3 most recent mistakes are always included regardless of ranking (a floor, not subject to the relevance score).

Writes happen two ways: explicitly (the `remember` tool) and automatically (`agent/session.py`'s end-of-session extraction pass, using a cheap model on the same provider as the main conversation). Both paths de-duplicate against existing memories — exact title match first, then a fuzzy Jaccard word-overlap fallback (`store.py`'s `find_duplicate`). `mazu memory consolidate` (`memory/consolidate.py`) does the same similarity check across the *whole* active set at once, clustering near-duplicates with a union-find over pairwise scores so transitive duplicates (A~B, B~C) merge correctly even when A and C aren't directly similar enough on their own.

## Checkpoints

A checkpoint (`checkpoint/manager.py`) bundles four things captured at the same instant:
1. A git commit (`git add -A && git commit --allow-empty`)
2. A consistent copy of `memory.db`, via SQLite's own online backup API — not a raw file copy, which could capture a half-written file if a transaction were in flight
3. A copy of the skills directory (gitignored, so `git clean` never touches it — needs its own snapshot/restore)
4. The conversation transcript at that point (`conversation.json`)

The index (`checkpoint/store.py`) is a flat, ordered JSON list, not a database — legible and diffable at the scale a single project's checkpoint history actually reaches. Checkpoint ids are always derived from the highest step number ever issued, not the current list length, so pruning old entries can never cause a new checkpoint to reuse (and silently corrupt) an id that's still referenced elsewhere.

Rollback is linear (`git reset --hard` + restore memory/skills + truncate the index after the restored point) — there's no branching tree yet (see `ROADMAP.md`). `mazu timeline` / `checkpoint show` / `checkpoint diff` are read-only inspection on top of the same index; none of them touch git state.

One git subtlety worth knowing if you're modifying this code: `git diff <commit> HEAD` compares two *commits*. If the checkpoint being diffed IS HEAD, that comparison is always empty even with real uncommitted changes — you have to omit the second ref entirely to diff a commit against the live working tree. `git diff` also never lists untracked files under any comparison, regardless of the refs involved; `checkpoint diff` supplements the raw git output with `git status --porcelain`'s untracked entries for exactly this reason.

## Checkpointing vs. memory vs. skills: why three separate mechanisms

They solve different problems and travel through the system differently on purpose:
- **Checkpoints** are point-in-time, rollback-able snapshots of *everything at once* — you restore to a moment.
- **Memory** is durable, project-level *knowledge* that should persist and be recalled across many future sessions — it isn't rolled back by a checkpoint restore to an *earlier* point unless that earlier checkpoint's own memory.db snapshot didn't have it yet.
- **Skills** are reusable *code* the agent wrote for itself — closer to a build artifact than either of the above, which is why they get their own snapshot/restore path instead of being folded into memory.
