# Mazu Roadmap

**Pitch:** Mazu is a local coding agent built around the layer *around* the model: what the agent remembers, what it did, whether you can undo it, what it cost, and whether a change to it actually helps. It is not an editor and does not try to replace one.

This file has three parts: what is **done**, what the project is **working toward now** (and how each item will be judged), and the **known gaps** that come first in the queue. Ideas that are deliberately not started are at the end.

## Done

The feature set below is implemented, tested, and documented in [README.md](README.md); [CHANGELOG.md](CHANGELOG.md) has version-by-version detail.

| Area | What exists |
|---|---|
| Checkpoints | Snapshot of git commit + memory database + skills + conversation; `timeline`, `checkpoint show / diff / inspect / compare / prune`, `rollback`, `branch-from`; branching runs (`run --from-checkpoint`, `checkpoint compare-branches`); cross-process index lock |
| Memory | Project and global SQLite stores, BM25 retrieval with optional semantic blend, pinned/mistake floor, `memory why / pin / edit / supersede / consolidate / stale / stats`, end-of-session extraction with de-duplication, retrieval tracking, observation-only belief logging (`memory-belief`) |
| Agent modes | `chat` (streaming, resume, compaction, auto-approve), `run` (unattended, resume, dry-run, step and cost limits), `explore` (parallel worktrees, test-based ranking), `council`, learning router suggestions, opt-in Curator |
| Safety | Root-confined file tools with atomic writes, shell denylist and opt-in allowlist with block reasons, confirmation prompts, long-running-server detection, action log (`mazu log`) |
| Providers | Anthropic, OpenAI, DeepSeek, Gemini, and local OpenAI-compatible servers behind one seam; normalized errors and retry; thread-safe client construction; model capability table |
| Cost | Usage log and `mazu usage`; `--max-cost` for run, chat, explore, and council, counting compaction and extraction; prompt-caching breakpoints for Anthropic |
| Measurement | `mazu bench`: repeatable task suites, fresh clone and throwaway HOME per trial, hidden-test verification, Wilson intervals, cost per solved task; two suites in `bench/` |
| UX | `mazu setup`, `mazu doctor [--live] [--fix]`, `mazu config`, `--json` output with a versioned envelope, terminal UI, [demo walkthrough](docs/demo.md) |

Not done from earlier plans: real streaming for Gemini (deferred until a live key can verify how the SDK delivers function-call chunks), structured per-file data for `checkpoint diff` (currently wraps `git diff --stat`), and an `examples/` directory.

## Working toward now

Each goal has a way to tell whether it worked; a change that does not move its number is dropped, not kept for its own sake.

1. **Cheaper models, closer to expensive-model results.** Raise the pass rate of a low-cost model on the benchmark and lower cost per solved task, using verification, sensible stopping, and cheaper context handling. *Judged by:* pass rate and cost per solved task on `mazu bench`, compared against a reference run with a stronger model on the same tasks (not yet run). The first baseline is in the README.
2. **Memory that earns its cost.** Keep what is useful without growth or junk, and without extra spend. Needs multi-session benchmark tasks, a trust/provenance field for auto-extracted entries, and a test of retrieval over raw history versus extracted entries. *Judged by:* task success with and without memory, memory cost per session, and the share of stored entries that are wrong or stale.
3. **Checkpoints you can trust under interruption.** Make `restore` robust to being interrupted partway. Step-boundary failures are now tested (and always recoverable by re-running); remaining: make an interrupted restore detect itself and finish on the next start, and test a real process kill mid-write. *Judged by:* tests that kill the process at each step and check that the next start leaves code, memory, skills, and index consistent.
4. **Evidence that survives scrutiny.** A benchmark with more than one repository, more repeats, and a head-to-head against other agents and models, so that claims about Mazu rest on numbers a stranger can reproduce.

## Known gaps, first in the queue

- `run_skill` is not confirmation-gated in `mazu run`, and skills run unsandboxed with Mazu's environment.
- The shell denylist is five patterns and the allowlist only checks how a command starts; neither parses shell syntax. A real answer is running unattended agents in a container or similar isolation.
- No verification step: the agent decides when it is finished, and was observed editing existing tests to make its own change pass. A `--protect-tests` prompt rule looked helpful in a pilot (19/25 versus 10/15) but did not hold up in a pre-registered confirmation run (24/35 versus 25/35 on real bugs, 24/27 versus 27/27 on injected bugs), so it stays an off-by-default experiment. The one real-bug task the agent never solves, `restore-open-connection` (0 of 22 trials), points at what is missing: a way to look at runtime state instead of only reading code.
- `restore` is a sequence of steps, not one transaction. Interrupting it between steps can leave code, memory, and skills out of step (4 of 7 steps), and running it again repairs that in every tested case (`tests/test_checkpoint_restore_interruption.py`). Still untested: a process killed mid-write.
- Auto-extracted memory is injected as established fact with no trust level, and contradictions are only logged (goal 2).
- The pricing table lags newer models; unpriced models cannot be cost-limited.
- mazu-web keeps its own copy of the chat loop, so loop-level improvements (such as chat `--max-cost`) do not reach its Chat tab; consolidating onto shared code is part of its planned UI rework.

## Directional (not committed)

- **Team/shared memory** (export/import, an approved-memory workflow): premature for a single-user tool.
- **Plugin/tool system:** premature without other contributors who want to extend it.
- **Editor extension:** designed in [docs/vscode-extension-design.md](docs/vscode-extension-design.md); implementation waits for evidence of demand and for structured diff output.
- **Usage metrics dashboard:** needs real usage data first.

## Versioning

No fixed version-to-feature mapping is promised. Each meaningful batch of work gets its own version bump; see the changelog and PyPI history for what shipped when. This file describes direction, not a release schedule.
