# Security Policy

## Reporting a vulnerability

Please **do not open a public GitHub issue** for a security vulnerability. Instead, email **turgut.sofuyev@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal example is ideal)
- Which version of Mazu you're running (`mazu --version`)

This is a solo-maintained open-source project, not a funded security team. You should get a response within a few days; please be patient.

## What Mazu does and does not protect against

Mazu runs on your own machine: the agent loop, tool execution, memory database, skill library, and checkpoints are all local. The only network traffic it generates is the model API call to the provider you configured, plus one cheap end-of-session call to that same provider for memory extraction. An agent that can edit files and run commands can do damage, and Mazu's protections are partial. Read this section before running it unattended.

**What exists:**

- **File tools** (`mazu/tools/fs.py`) are confined to the project root, including through symlinks.
- **Confirmation.** In `mazu chat`, every destructive tool call (file writes and edits, shell commands, running a skill) asks first unless you enable auto-approve. In `mazu run`, shell commands ask unless `--allow-shell` is passed.
- **A shell denylist** (`mazu/tools/shell.py`) blocks five patterns: `rm -rf /`, `git push --force`, paths under `.ssh/`, `sudo`, and `format X:`.
- **`--shell-allowlist`**, opt-in, restricts commands to ones that start with a listed program name.
- **`--dry-run`** previews a run without writing files or executing commands.
- **Checkpoints** can undo file and memory changes.
- **API keys** are read from environment variables or `~/.mazu/config.toml` (in your home folder, outside any repo) and are masked in `mazu config list`. Mazu adds `.mazu/` to a project's `.gitignore` but does not manage `.env` files.

**What does not exist (do not rely on these):**

- **The denylist is not a sandbox.** It is a few regular expressions and is bypassed by equivalent commands (for example `rm -rf ~`, `git push -f`, `su`, PowerShell deletions, or a Python one-liner), by commands that touch files outside the project, and by anything not on the list.
- **The allowlist only checks the start of a command.** A compound command such as `git status; other-command` passes an allowlist of `git`.
- **Skills are arbitrary Python** executed as a subprocess with the same privileges and environment variables as Mazu itself, including any API keys in the environment, and no isolation. In `mazu run`, saved skills are executed **without a confirmation prompt** and `--allow-shell` does not govern them; in `mazu chat` they ask first. A skill written by the agent, or one you did not review, is equivalent to running arbitrary code.
- **Checkpoints cannot undo external actions**, such as a push, a sent request, or anything outside the project directory and Mazu's own state.
- **Memory is trusted.** Facts extracted automatically from a session are stored and later shown to the model as established context. Content the agent read (a file, a command's output) can end up in memory, and Mazu does not verify or label it. If you work with untrusted repositories, review `mazu memory list` and use `mazu memory forget` on anything wrong.
- **`restore` is not atomic.** If a rollback is interrupted partway, code may be restored while memory, skills, or the checkpoint index are not.

**Practical advice:** run unattended agents (`mazu run --allow-shell`, `mazu explore`) in a disposable directory, a throwaway clone, or a container; set `--max-cost`; keep API keys scoped and low-limit; review `mazu log show <session>` afterwards.

## Known, accepted limitations (not bugs to report)

These are documented tradeoffs and are tracked in [ROADMAP.md](ROADMAP.md):

- A shell command's working directory is a starting point, not a boundary: a single approved command can `cd ..` out of it.
- The denylist and allowlist are not exhaustive, as described above.
- Skills have no extra sandboxing, and `mazu run` does not gate them.
- Memory has no trust levels.

If you believe one of these should be treated differently, open a regular issue to discuss the design. A concrete bypass of a protection that is *claimed* above (for example, a path that escapes the root-confined file tools) is a vulnerability: report it privately.

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/mazu/) is supported. Please upgrade (`pip install --upgrade mazu`) and confirm the problem still reproduces before reporting.
