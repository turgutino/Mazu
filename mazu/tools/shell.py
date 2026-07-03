import re
import subprocess
from pathlib import Path

from mazu.tools.base import Tool, ToolResult

# Hardcoded backstop regardless of confirmation or --allow-shell: checkpoints undo
# file damage, not irreversible external actions (force-pushes, key exfiltration,
# disk wipes). Shared between chat mode (loop.py) and autonomous mode
# (autonomous.py) so a reflexive "y" in chat isn't the only thing standing between
# the model and one of these.
SHELL_DENYLIST = [
    re.compile(r"rm\s+-rf\s+/(\s|$)", re.IGNORECASE),
    re.compile(r"git\s+push\b.*--force", re.IGNORECASE),
    re.compile(r"\.ssh(/|\\)", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
]


def is_denied_shell_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in SHELL_DENYLIST)


def make_shell_tool(root: Path, timeout: int = 60) -> Tool:
    def run_shell(input: dict) -> ToolResult:
        command = input["command"]
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
        description="Run a shell command in the project root and return its stdout/stderr/exit code.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        handler=run_shell,
        destructive=True,
    )
