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
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
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
            "instead of raising an error, so don't assume Unix flags work here."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        handler=run_shell,
        destructive=True,
    )
