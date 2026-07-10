# Security Policy

## Reporting a vulnerability

Please **do not open a public GitHub issue** for a security vulnerability. Instead, email **turgut.sofuyev@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal example is ideal)
- Which version of Mazu you're running (`mazu --version`)

You should get a response within a few days. This is a solo-maintained open-source project, not a funded security team — please be patient, but you will be heard.

## Scope

Mazu runs entirely on your own machine: the agent loop, tool execution, memory database, skill library, and checkpoints are all local. The only network traffic it generates is the model API call to whichever provider you've configured, plus one cheap end-of-session call (to the same provider) for memory extraction. Relevant attack surface:

- **File tool sandboxing** (`mazu/tools/fs.py`) — `read_file`/`write_file`/`edit_file`/`glob_files` are meant to be confined to the project root, including through symlinks.
- **Shell command execution** (`mazu/tools/shell.py`) — gated by confirmation (unless `--allow-shell`) and a hardcoded denylist for a short list of genuinely destructive patterns (force-push, `sudo`, `~/.ssh`, disk-format commands, `rm -rf /`). This denylist is a backstop for obviously dangerous commands, **not** a full sandbox — a sufficiently creative command can still do damage within the project directory, same as any comparable tool's shell-exec feature.
- **Skill execution** (`mazu/skills/manager.py`) — a saved skill is arbitrary Python code, executed as a local subprocess. Skills are written by the agent for your own project, but running a skill you didn't personally review (e.g. one shared by someone else) is equivalent to running arbitrary code — treat it accordingly.
- **API keys** — only ever read from environment variables or `~/.mazu/config.toml` (outside any repo). Mazu never writes a key to disk itself.

## Known, accepted limitations (not bugs to report)

These are documented tradeoffs, not vulnerabilities:

- `run_shell`'s working directory is a starting point, not a real sandbox against `cd ..`-style escapes within a single approved command.
- The shell denylist is a short list of obviously dangerous patterns, not an exhaustive one.
- Skills execute with the same privileges as the `mazu` process itself (no additional sandboxing).

If you believe one of these should be treated differently, that's a reasonable design discussion — open an issue (not a private report) to discuss it.

## Supported versions

Only the latest released version on [PyPI](https://pypi.org/project/mazu/) is supported. Please upgrade (`pip install --upgrade mazu`) before reporting an issue to confirm it still reproduces.
