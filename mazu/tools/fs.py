from pathlib import Path

from mazu.tools.base import Tool, ToolResult


def _safe_path(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path '{path}' escapes the project root")
    return resolved


def make_fs_tools(root: Path) -> list[Tool]:
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
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(input["content"], encoding="utf-8")
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
            p.write_text(text.replace(old, new), encoding="utf-8")
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
