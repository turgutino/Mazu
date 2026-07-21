import os
import tempfile
from pathlib import Path

from mazu.tools.base import Tool, ToolResult


def _safe_path(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path '{path}' escapes the project root")
    return resolved


def _atomic_write_text(path: Path, content: str) -> None:
    """Writes `content` to `path` crash-safely: if the process (or the machine) dies
    mid-write, the original file (if any) is left fully intact, and the new file is
    never left half-written at the real path. A plain `Path.write_text` is a stream
    of smaller writes with no such guarantee -- a crash partway through can leave a
    truncated or corrupted file with no way to tell it apart from a real one.

    Mechanism: write the full content to a temp file in the SAME directory as the
    target (same filesystem is required for the final rename to be atomic), flush
    and fsync it to disk, then `os.replace()` it onto the real path -- a single
    filesystem-level rename that either fully happens or fully doesn't, on both
    Windows and POSIX. There is no in-between state a crash can observe.
    """
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def make_fs_tools(root: Path, dry_run: bool = False) -> list[Tool]:
    def read_file(input: dict) -> ToolResult:
        try:
            p = _safe_path(root, input["path"])
            if not p.exists():
                return ToolResult(f"File not found: {input['path']}", is_error=True)
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            numbered = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))
            return ToolResult(numbered or "(empty file)")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    def write_file(input: dict) -> ToolResult:
        try:
            p = _safe_path(root, input["path"])
            if dry_run:
                return ToolResult(f"[dry-run] Would write {len(input['content'])} bytes to {input['path']}")
            p.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(p, input["content"])
            return ToolResult(f"Wrote {len(input['content'])} bytes to {input['path']}")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    def edit_file(input: dict) -> ToolResult:
        try:
            p = _safe_path(root, input["path"])
            text = p.read_text(encoding="utf-8")
            old, new = input["old_str"], input["new_str"]
            count = text.count(old)
            if count == 0:
                return ToolResult("old_str not found in file", is_error=True)
            if count > 1:
                return ToolResult(
                    f"old_str matches {count} times; must be unique", is_error=True
                )
            # Even in dry-run mode, the file is still read and old_str's match count
            # validated above -- this is what makes the dry-run report trustworthy
            # (a plan that would fail for real fails here too, not just in the
            # eventual real run) rather than an unconditional "sure, would work."
            if dry_run:
                return ToolResult(f"[dry-run] Would edit {input['path']} (replacing 1 occurrence)")
            _atomic_write_text(p, text.replace(old, new))
            return ToolResult(f"Edited {input['path']}")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    def list_dir(input: dict) -> ToolResult:
        try:
            p = _safe_path(root, input.get("path", "."))
            entries = sorted(
                x.name + ("/" if x.is_dir() else "") for x in p.iterdir()
            )
            return ToolResult("\n".join(entries) if entries else "(empty directory)")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    def glob_files(input: dict) -> ToolResult:
        try:
            # root.glob() matches by path string, not resolved target -- a symlink
            # inside the project pointing outside it would otherwise let a pattern
            # like "escape_link/*" return files outside the sandbox. Resolve each
            # match and apply the same boundary check _safe_path uses elsewhere.
            matches = []
            for x in root.glob(input["pattern"]):
                resolved = x.resolve()
                if resolved != root and root not in resolved.parents:
                    continue
                matches.append(str(x.relative_to(root)))
            matches.sort()
            return ToolResult("\n".join(matches) if matches else "(no matches)")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

    return [
        Tool(
            name="read_file",
            description="Read a file's contents, returned with line numbers.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=read_file,
        ),
        Tool(
            name="write_file",
            description="Create or overwrite a file with the given content.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=write_file,
            destructive=True,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact, unique occurrence of old_str with new_str in a file. "
                "Fails if old_str appears zero or more than once."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
            handler=edit_file,
            destructive=True,
        ),
        Tool(
            name="list_dir",
            description="List files and directories at a path (default: project root).",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
            handler=list_dir,
        ),
        Tool(
            name="glob_files",
            description="Find files matching a glob pattern relative to the project root.",
            input_schema={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
            handler=glob_files,
        ),
    ]
