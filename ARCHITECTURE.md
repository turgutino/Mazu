# Architecture

System-design detail split out of [README.md](README.md), which stays focused on using Mazu. Read this if you are contributing, debugging, or want to know where a behavior lives. For what is *not* guaranteed, read [Known structural caveats](#known-structural-caveats) at the end.

## Module map

```
mazu/
├── cli.py              Click entry point: every command and flag (chat, run, explore, council, checkpoint, memory,
│                       skills, curator, router, log, usage, bench, config, doctor, setup, ui, ...)
├── config.py           ~/.mazu/config.toml read/write; model resolution (env > config > detected provider)
├── diagnostics.py      `mazu doctor`: Python/git checks, key presence, live key test, safe auto-fixes (.gitignore, git init)
├── output.py           the single --json envelope: {"schema_version", "mazu_version", "data"}
├── banner.py           startup banner
│
├── agent/
│   ├── loop.py             interactive chat REPL: streaming, /checkpoint /rollback /compact /auto, --max-cost
│   ├── autonomous.py       unattended `mazu run`: step loop, circuit breaker, cost limit, compaction, dry-run, resume/fork
│   ├── explore.py          `mazu explore`: N branches on isolated git worktrees, ranking by tests/cost
│   ├── council.py          `mazu council`: parallel read-only advisory round + lead synthesis; thread-safe shared cost cap
│   ├── compaction.py       summarizes the oldest part of a long conversation to stay inside the context budget
│   ├── context.py          assembles the system prompt: base prompt + global memory + project memory + skills
│   ├── prompts.py          the base system prompt (including Mazu's own identity disclosure)
│   ├── session.py          end-of-session / mid-compaction memory extraction and insertion
│   ├── registry_factory.py builds the tool registry shared by chat, run, council and explore
│   └── interaction.py      safe_confirm(): EOF/interrupt-safe y/N prompts
│
├── llm/
│   ├── client.py           the single seam every model call goes through (run_turn / run_turn_stream / run_forced_tool)
│   ├── providers/          Anthropic, OpenAI, DeepSeek, Gemini, local (OpenAI-compatible) adapters behind one Provider ABC
│   ├── errors.py           normalized error hierarchy (rate limit, auth, transient, context length)
│   ├── error_mapping.py    maps SDK exceptions onto that hierarchy
│   ├── retry.py            exponential backoff for retryable errors only
│   ├── pricing.py          approximate per-model prices for --max-cost / `mazu usage` (can lag provider pricing)
│   ├── capabilities.py     streaming / tool-use / context-window table behind `mazu models`
│   └── types.py            AgentResponse and related types
│
├── memory/
│   ├── store.py            SQLite memory store (one class backs both the project and the global store)
│   ├── retrieval.py        BM25 ranking (+ optional semantic blend), pinned/mistake floor, context-block rendering
│   ├── bm25.py             pure-Python Okapi BM25, no dependencies
│   ├── embeddings.py       optional semantic layer (opt-in via MAZU_SEMANTIC_MEMORY)
│   ├── extraction.py       the cheap-model extraction prompt (at most 5 memories per call)
│   ├── consolidate.py      near-duplicate clustering and superseding (`mazu memory consolidate`)
│   ├── staleness.py        detection of stale memories (`mazu memory stale`)
│   ├── belief.py           SHADOW-MODE contradiction classifier: logs what it would do, never mutates memory
│   ├── belief_log.py       append-only log behind `mazu memory-belief`
│   └── schema.sql          memories / sessions tables
│
├── checkpoint/
│   ├── manager.py          snapshot / restore / fork / branch_from / timeline / diff / prune
│   ├── store.py            the checkpoint index: a flat JSON list (human-readable, diffable)
│   ├── lock.py             cross-process file lock (filelock) around index updates and git calls
│   └── worktree.py         git worktree lifecycle for `mazu explore`
│
├── tools/                  what the model can call
│   ├── fs.py               read_file, write_file, edit_file (atomic temp-then-replace), list_dir, glob_files; root-confined
│   ├── shell.py            run_shell, the denylist, the allowlist check, long-running-server detection
│   ├── memory_tools.py     remember, recall
│   ├── skill_tools.py      save_skill, list_skills, run_skill
│   ├── registry.py, base.py
│
├── skills/manager.py       save / list / run / archive local skill functions (subprocess, args via stdin)
├── runs/                   run bookkeeping (store.py), router suggestions from explore history (router.py),
│                           test-command detection (detect_test_command.py)
├── chat/store.py          persisted chat transcripts (`mazu chat --resume`)
├── action_log/store.py     every tool call: input, output, outcome (ok / error / blocked / declined / unknown_tool)
├── usage/store.py          cross-project spend log
├── curator/                opt-in autonomous maintenance of Mazu's own state, with its own model and key
│                           (areas: memory, skills, housekeeping, conflicts; every change logged and reversible)
├── bench/                  `mazu bench`: spec.py (suite/config files), runner.py (trials), stats.py (Wilson intervals, report)
├── files/document_extract.py  .docx/.pdf/.xlsx/.pptx to text (used by mazu-web attachments)
└── ui/                     Textual terminal UI over checkpoints, memory, and the action log

bench/                      benchmark suite builders (outside the package): mazu_history/ and seeded_bugs/
```

## Where state lives

Project-scoped, inside `<project>/.mazu/` (gitignored): `memory.db`, `action_log.db`, `runs.db`, `chat_history.db`, `belief_shadow.db`, `curator.db`, `skills/`, `checkpoints/` (index plus one folder per checkpoint).
Global, inside `~/.mazu/`: `config.toml`, `global_memory.db` (user-preference facts), `usage.db` (spend across all projects).

## The provider seam

Every model call goes through `mazu/llm/client.py` (`run_turn`, `run_turn_stream`, `run_forced_tool`). Nothing else imports `anthropic`, `openai`, or `google.genai`. To add a provider:

1. Implement the `Provider` ABC (`mazu/llm/providers/base.py`): `run_turn`, `run_forced_tool`, optionally `run_turn_stream`.
2. Convert between Mazu's canonical Anthropic-shaped messages/tools and the SDK's own format (see `gemini_provider._to_gemini_contents` for the kind of mismatch to expect).
3. Register it in `client.py`'s `_PROVIDERS` and default-model table.

Provider instances are process-wide singletons shared across threads (mazu-web runs each session on its own thread), so lazy SDK-client construction goes through `Provider._get_or_create_client()`, which uses double-checked locking. Errors are normalized so callers only catch `MazuAPIError` and its subclasses. A missing API key can surface from some SDKs only at request time as a bare `TypeError`; `error_mapping.classify_sdk_error` turns that into `MazuAuthError`.

## Tool execution and where it is gated

- **Chat (`agent/loop.py`):** every tool marked `destructive` (file writes and edits, `run_shell`, `run_skill`) asks for confirmation unless auto-approve is on. For `run_shell`, the denylist and any `--shell-allowlist` are checked first.
- **Run (`agent/autonomous.py`):** the denylist and allowlist are checked for `run_shell`, which then asks unless `--allow-shell`. **Other destructive tools, including `run_skill`, are not gated here** (file edits are meant to proceed unattended, protected by checkpoints). See the caveats.
- The denylist (`shell.SHELL_DENYLIST`) is five regular expressions; the allowlist compares only the start of the command string. Neither parses shell syntax.

## Memory

Two `MemoryStore` instances per session, same class, different files: project (`.mazu/memory.db`: decision, convention, mistake, task_outcome, fact) and global (`~/.mazu/global_memory.db`: `user_preference`).

- **Writes:** explicitly via the `remember` tool, and automatically via `session.insert_extracted_memories` (end of session, and on each compacted chunk). Both paths de-duplicate by exact normalized title, then by word-overlap similarity within the category (`MemoryStore.find_duplicate`). Extraction asks for at most five memories.
- **Reads:** `retrieval.build_context_block` takes pinned memories and the three most recent mistakes unconditionally, ranks the rest with BM25 (plus cosine similarity if semantic search is on), takes at most 15 ranked ones, and renders them within an 8,000-character budget (the floor is exempt from the budget). Only memories actually rendered have their `retrieval_count` and `last_used_at` updated. The rendered block tells the model to treat memory as ground truth unless the task contradicts it.
- **Not implemented:** per-memory trust level, automatic contradiction resolution (the belief classifier only logs), automatic pruning (only `consolidate`, `stale`, `forget`, and Curator remove or retire entries), and any quality benchmark for retrieval.

## Checkpoints

`CheckpointManager.snapshot` holds the index lock for its whole body and, in order: runs `git add -A` and `git commit --allow-empty`, reads the commit hash, then writes `conversation.json`, an online-backup copy of `memory.db` (SQLite backup API, safe mid-write), and a copy of the skills folder, and finally appends the index entry and prunes. A failure before the index append removes the partial folder. The git commit is not part of that cleanup: a hard crash between the commit and the index append leaves a commit the index does not know about (documented in the source).

`restore` runs, in order: `git reset --hard`, `git clean -fd`, memory-database restore, skills restore (delete then copy), read `conversation.json`, truncate the index. These are separate steps, **not one transaction**. `fork` is the non-destructive counterpart: a new branch at the checkpoint, with memory and skills restored onto it and the origin branch's index entries left alone.

Index entries carry `branch` and `parent_checkpoint_id` (additive, older entries lack them) so timeline, prune, and no-argument rollback are branch-correct. Checkpoint ids come from the highest step ever issued, not the list length, so pruning never causes id reuse.

A git subtlety if you touch this code: `git diff <commit> HEAD` compares two commits and is empty when the checkpoint *is* HEAD; to diff against the live tree, omit the second ref. `git diff` never lists untracked files, so `checkpoint diff` appends `git status --porcelain` entries.

## Cost control

`usage/store.py` logs every call (provider, model, tokens, estimated cost) per session id. `--max-cost` for `mazu run` and `mazu chat` sums the main loop **plus** compaction and memory-extraction calls, reading all of them through the same tracker; `explore` and `council` share one thread-safe cap across parallel members. Limits are checked after each round, so the round that crosses the limit is still charged. A model with no entry in `pricing.py` is reported as unpriced and the limit cannot be enforced for it.

## Benchmarking (`mazu bench`)

`bench/spec.py` loads a suite (tasks: repo, optional ref, prompt, verify command, step cap) and configs (name, model, extra `mazu run` args, env). `bench/runner.py` runs each trial in a fresh `git clone --local` with a throwaway HOME (config copied in for keys), runs the *installed* Mazu (the cloned code is scrubbed from `sys.path`), then decides pass/fail with the verify command, never the agent's claim. Cost comes from that trial's own usage rows. Results are append-only JSONL, interleaved across configs, resumable. `bench/stats.py` computes Wilson 95% intervals and cost per solved task. The suite builders in `bench/` prove each task is fair (`--check`): the unfixed code must fail the hidden test and the reference fix must pass it.

## Why checkpoints, memory, and skills are separate

- **Checkpoints** are point-in-time snapshots of everything at once: you restore to a moment.
- **Memory** is durable project knowledge recalled across sessions; a restore to an earlier checkpoint also restores that checkpoint's copy of the memory database.
- **Skills** are reusable code the agent wrote, closer to a build artifact, which is why they have their own snapshot and restore path.

## Known structural caveats

- **Restore is sequential, not atomic.** An interruption between its steps can leave code rolled back while memory, skills, or the index are not. There is no fault-injection test for `restore` (only a snapshot-failure test exists).
- **`run_skill` is ungated in `mazu run`**, and skills run with Mazu's privileges and environment without isolation.
- **The shell denylist and allowlist are string matching**, not a sandbox.
- **mazu-web duplicates the chat loop** (`mazu_web/chat_session.py`) instead of calling `agent/loop.py`, so chat-loop changes here (such as `--max-cost`) do not reach its Chat tab automatically; its Run, Explore, and Council tabs call the same functions as the terminal.
- **No verification step in the agent loop:** the model decides when it is finished.
- **Memory content is trusted** once stored, and extraction is lossy by design.
