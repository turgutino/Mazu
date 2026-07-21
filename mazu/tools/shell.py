import os
import platform
import re
import subprocess
from pathlib import Path

from mazu.tools.base import Tool, ToolResult

# subprocess.run(shell=True) always goes through cmd.exe on Windows (via %COMSPEC%),
# regardless of what shell the user's own terminal happens to be — so this is an
# accurate, not just a best-guess, description of what actually executes the command.
_SHELL_LABEL = "Windows cmd.exe" if platform.system() == "Windows" else "a POSIX shell (sh)"

# Hardcoded backstop regardless of confirmation or --allow-shell: checkpoints undo
# file damage, not irreversible external actions (force-pushes, key exfiltration,
# disk wipes). Shared between chat mode (loop.py) and autonomous mode
# (autonomous.py) so a reflexive "y" in chat isn't the only thing standing between
# the model and one of these. Each entry pairs its pattern with a short, human-
# readable reason so a block message can say *why*, not just "denylist matched" --
# a model (and a user reading the transcript) can act on "elevates privileges via
# sudo" in a way it can't act on an opaque regex match.
SHELL_DENYLIST: list[tuple[re.Pattern, str]] = [
    (re.compile(r"rm\s+-rf\s+/(\s|$)", re.IGNORECASE), "recursively deletes the filesystem root"),
    (re.compile(r"git\s+push\b.*--force", re.IGNORECASE), "force-pushes, which can overwrite others' history"),
    (re.compile(r"\.ssh(/|\\)", re.IGNORECASE), "touches your SSH credentials directory"),
    (re.compile(r"\bsudo\b", re.IGNORECASE), "elevates privileges via sudo"),
    (re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE), "formats a drive"),
]


def denylist_reason(command: str) -> str | None:
    """Returns a human-readable reason the command is blocked, or None if it isn't
    denylisted. The first matching rule wins if more than one pattern matches."""
    for pattern, reason in SHELL_DENYLIST:
        if pattern.search(command):
            return reason
    return None


def is_denied_shell_command(command: str) -> bool:
    return denylist_reason(command) is not None


# Real problem found via live testing: the model tried to "test" a Flask dev server
# with `python app.py` via run_shell -- which waits for a command to FINISH (or hit
# its timeout) before returning control. A dev server never exits on its own, so this
# just blocks for the full timeout (or until the user manually interrupts the whole
# mazu process with Ctrl+C, which is what actually happened -- confirmed live). This
# is not a safety concern like SHELL_DENYLIST above (nothing dangerous happens), so it
# doesn't belong in that list -- it's a "this can't work the way you're trying to use
# it" case, caught before ever starting the subprocess rather than after blocking.
# Necessarily a heuristic (matches common dev-server invocations, not exhaustive) --
# false negatives just fall through to the normal timeout behavior, no worse than
# before this existed.
LONG_RUNNING_SERVER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bflask\s+run\b", re.IGNORECASE), "starts a Flask development server"),
    (re.compile(r"\.run\s*\(", re.IGNORECASE), "calls a server's .run(...) entry point (e.g. Flask/Django app.run())"),
    (re.compile(r"manage\.py\s+runserver\b", re.IGNORECASE), "starts a Django development server"),
    (re.compile(r"\bnpm\s+(run\s+)?(dev|start)\b", re.IGNORECASE), "starts an npm dev/start server"),
    (re.compile(r"\byarn\s+(dev|start)\b", re.IGNORECASE), "starts a yarn dev/start server"),
    (re.compile(r"\bvite\b(?!\s+build)", re.IGNORECASE), "starts a Vite dev server"),
    (re.compile(r"\bnext\s+dev\b", re.IGNORECASE), "starts a Next.js dev server"),
    (re.compile(r"\b(uvicorn|gunicorn)\b", re.IGNORECASE), "starts an ASGI/WSGI application server"),
    (re.compile(r"python\s+-m\s+http\.server\b", re.IGNORECASE), "starts a simple HTTP server"),
    (re.compile(r"\brails\s+s(erver)?\b", re.IGNORECASE), "starts a Rails development server"),
]


_BARE_PYTHON_SCRIPT_RE = re.compile(
    r"^\s*(python3?|py)\s+([\w./\\-]+\.py)\s*$", re.IGNORECASE
)

# The command line alone often gives NO hint at all: the real case that motivated
# this (`python app.py`) looks identical whether app.py prints "hello" and exits or
# starts a Flask server that never returns -- that information lives inside the
# script's own source, not the shell command invoking it. So for the common "python
# <file>.py" shape specifically, also peek at the target file's own content for
# well-known server-starting calls, rather than relying on command-line text alone.
_SERVER_SOURCE_SIGNATURES: list[tuple[str, str]] = [
    # Deliberately specific (not a bare ".run(") -- a generic ".run(" substring
    # search would false-positive on plenty of legitimate, short-lived scripts that
    # happen to call subprocess.run(...), a threading.Thread(...).run(), an asyncio
    # event loop's run_until_complete(...), etc. These four are the actual,
    # well-known "this starts a server" call shapes for the frameworks a Mazu user
    # is realistically writing (Flask/Dash-style .run(), Django dev server helper,
    # Python's built-in socket/HTTP servers, and uvicorn for ASGI apps).
    ("app.run(", "calls app.run(...) (e.g. Flask/Dash)"),
    (".run(host=", "calls .run(host=...) to bind a server to an address"),
    (".run(debug=", "calls .run(debug=...), the common Flask dev-server invocation"),
    ("serve_forever(", "calls serve_forever(...) (e.g. Python's built-in HTTP/socket servers)"),
    ("uvicorn.run(", "calls uvicorn.run(...) to start an ASGI server"),
]


def _bare_python_script_target(command: str, root: Path) -> Path | None:
    """If `command` is exactly a bare `python <file>.py` invocation (no extra args,
    no shell operators), returns the resolved, root-confined path to that script --
    the one shape simple enough that peeking at its source is worth the extra file
    read. Returns None for anything more complex (compound commands, extra
    arguments, a path escaping root, or a file that doesn't exist).
    """
    match = _BARE_PYTHON_SCRIPT_RE.match(command)
    if match is None:
        return None
    try:
        resolved = (root / match.group(2)).resolve()
    except OSError:
        return None
    if resolved != root and root not in resolved.parents:
        return None
    return resolved if resolved.is_file() else None


def long_running_server_reason(command: str, root: Path) -> str | None:
    """Returns a human-readable reason `command` looks like it starts a long-running
    server/dev process that never exits on its own, or None if it doesn't match any
    known pattern. Checked before running anything, so the model gets an immediate,
    clear answer instead of a blocked subprocess burning the full timeout. Checks the
    command line itself against known dev-server invocations, and additionally peeks
    at the target file's own source for a bare `python <file>.py` command, since the
    command line alone doesn't say what the script does once it starts running.
    """
    for pattern, reason in LONG_RUNNING_SERVER_PATTERNS:
        if pattern.search(command):
            return reason

    script_path = _bare_python_script_target(command, root)
    if script_path is not None:
        try:
            source = script_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            source = ""
        for signature, reason in _SERVER_SOURCE_SIGNATURES:
            if signature in source:
                return f"{reason} in {script_path.name}"
    return None


def is_allowed_by_shell_allowlist(command: str, allowlist: list[str] | None) -> bool:
    """An allowlist is opt-in and additive to the denylist above, not a replacement
    for it -- the denylist backstop always applies regardless of allowlist state. With
    no allowlist configured (the default), every non-denylisted command is allowed,
    unchanged from before this existed. When an allowlist IS configured, a command is
    only allowed if it starts with one of the allowed program names as a whole word
    (e.g. allowlist=["git"] permits "git status" but not "gitx status" or a command
    that merely mentions "git" later, like "echo git").
    """
    if not allowlist:
        return True
    stripped = command.strip()
    return any(re.match(rf"{re.escape(name)}\b", stripped) is not None for name in allowlist)


def make_shell_tool(root: Path, timeout: int = 60, dry_run: bool = False) -> Tool:
    def run_shell(input: dict) -> ToolResult:
        command = input["command"]
        if dry_run:
            return ToolResult(f"[dry-run] Would run: {command}")
        server_reason = long_running_server_reason(command, root)
        if server_reason is not None:
            return ToolResult(
                f"Not run: this command {server_reason}, which never exits on its own. "
                "run_shell waits for a command to finish (or times out) before returning "
                "control, so a persistent server would just block until then. Ask the "
                "user to start this in their own terminal instead, and verify the result "
                "another way (e.g. checking the files you changed, or asking the user "
                "what they see when they open it themselves).",
                is_error=True,
            )
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=root,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                # Real bug found via live testing (twice, independently -- once with
                # an emoji in a generated print statement, once with a Turkish/
                # Azerbaijani letter): on Windows, a spawned Python script's own
                # stdout defaults to the console's legacy codepage (cp1252), not
                # UTF-8, so printing non-ASCII text crashes the *subprocess* with
                # its own UnicodeEncodeError -- a failure the model then has to
                # notice and fix in a follow-up round, wasting a step. Setting
                # PYTHONIOENCODING here fixes it at the source for any Python
                # subprocess `run_shell` launches, on top of `encoding`/`errors`
                # above, which only make *our own* capture of stdout/stderr safe,
                # not the subprocess's own internal print() calls.
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            )
            output = proc.stdout
            if proc.stderr:
                output += "\n--- stderr ---\n" + proc.stderr
            output += f"\n--- exit code: {proc.returncode} ---"
            return ToolResult(output, is_error=proc.returncode != 0)
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or "") + (f"\n--- stderr ---\n{e.stderr}" if e.stderr else "")
            return ToolResult(
                f"Command timed out after {timeout}s.\n{partial}".rstrip(), is_error=True
            )
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    return Tool(
        name="run_shell",
        description=(
            "Run a shell command in the project root and return its stdout/stderr/exit code. "
            f"Commands execute via {_SHELL_LABEL} — use syntax native to that shell, not "
            "another OS's. For example, on Windows cmd.exe, `mkdir` already creates "
            "intermediate directories on its own and does not understand a Unix-style `-p` "
            "flag; passing one silently creates a stray directory literally named `-p` "
            "instead of raising an error, so don't assume Unix flags work here. "
            "This waits for the command to finish (or times out) before returning -- it "
            "cannot be used to start a dev server or any other long-running process that "
            "never exits on its own (e.g. `flask run`, `npm run dev`, `python app.py` "
            "calling `app.run()`); ask the user to start those themselves in their own "
            "terminal instead."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        handler=run_shell,
        destructive=True,
    )
