"""Best-effort detection of a project's test command, used to pre-fill
Explore's optional "Test command" field with a suggestion instead of leaving
the user to guess or type it from scratch. Purely a suggestion: the caller
decides whether to show/apply it, and this module never executes anything
itself -- only file-system reads (existence checks, small text reads). A wrong
guess costs nothing worse than a failed/no-op shell command if it's actually
used later; test commands are read-only by nature, they don't mutate the
project being checked.
"""

import json
import platform
from pathlib import Path

# Directories never worth descending into: dependency/build/VCS noise that
# could be huge (node_modules) or contain a misleading nested project (a
# vendored copy of another package with its own go.mod/package.json).
_IGNORED_DIR_NAMES = {
    ".git", ".mazu", ".hg", ".svn", "node_modules", "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", "dist", "build", ".next", "target", "vendor",
}

# How many directory levels below the project root to search -- covers the
# common "the real app lives one level down" case (e.g. a `backend/` or
# `magaza/` subdirectory) without scanning an entire large monorepo.
_MAX_DEPTH = 2

_IS_WINDOWS = platform.system() == "Windows"


def _venv_python_relative(directory: Path) -> str | None:
    """Returns a path (relative to `directory`, using this OS's own separator)
    to a virtualenv's python executable if one of the conventional venv
    directory names exists directly inside `directory`, else None.
    """
    bin_dir = "Scripts" if _IS_WINDOWS else "bin"
    exe_name = "python.exe" if _IS_WINDOWS else "python"
    sep = "\\" if _IS_WINDOWS else "/"
    for venv_name in ("venv", ".venv", "env"):
        exe = directory / venv_name / bin_dir / exe_name
        if exe.exists():
            return f"{venv_name}{sep}{bin_dir}{sep}{exe_name}"
    return None


def _candidate_dirs(root: Path):
    """Yields `root` itself, then its subdirectories breadth-first up to
    _MAX_DEPTH levels deep, skipping noise directories -- root first so a
    project with its own marker files at the top level is never shadowed by
    an unrelated nested one, and shallower matches always win over deeper ones.
    """
    yield root
    queue = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth >= _MAX_DEPTH:
            continue
        try:
            children = sorted(
                p for p in current.iterdir()
                if p.is_dir() and p.name not in _IGNORED_DIR_NAMES and not p.name.startswith(".")
            )
        except OSError:
            continue
        for child in children:
            yield child
            queue.append((child, depth + 1))


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _suggest_for_dir(directory: Path) -> str | None:
    """Returns a bare command (no `cd` prefix) for `directory` if it recognizes
    a project marker there, else None. Checked in a fixed priority order --
    the first match wins, since a real directory rarely carries markers for
    more than one language/ecosystem.
    """
    has_pytest_ini = (directory / "pytest.ini").exists()
    has_pyproject_pytest = "[tool.pytest" in _read_text_safe(directory / "pyproject.toml")
    has_setup_cfg_pytest = "[tool:pytest]" in _read_text_safe(directory / "setup.cfg")
    tests_dir = directory / "tests"
    has_tests_dir = tests_dir.is_dir() and any(tests_dir.glob("test_*.py"))
    if has_pytest_ini or has_pyproject_pytest or has_setup_cfg_pytest or has_tests_dir:
        python_exe = _venv_python_relative(directory) or "python"
        return f"{python_exe} -m pytest -q"

    package_json = directory / "package.json"
    if package_json.exists():
        try:
            data = json.loads(_read_text_safe(package_json) or "{}")
            if isinstance(data.get("scripts"), dict) and "test" in data["scripts"]:
                return "npm test"
        except json.JSONDecodeError:
            pass

    if (directory / "go.mod").exists():
        return "go test ./..."

    if (directory / "Cargo.toml").exists():
        return "cargo test"

    if (directory / "pom.xml").exists():
        return "mvn test"

    if (directory / "build.gradle").exists() or (directory / "build.gradle.kts").exists():
        return "gradlew.bat test" if _IS_WINDOWS else "./gradlew test"

    return None


def detect_test_command(root: Path) -> str | None:
    """Best-effort suggestion for Explore's optional test command, or None if
    nothing recognizable was found within `_MAX_DEPTH` levels of `root`.
    Searches breadth-first (root first, then its subdirectories), skipping
    common noise directories (node_modules, venv, .git, etc.) that could be
    huge or misleading.

    The returned command is prefixed with `cd <relative-dir> && ` when the
    match isn't at `root` itself, since a shell command's cwd is always the
    project root, not wherever the marker file happened to be found.
    """
    for directory in _candidate_dirs(root):
        suggestion = _suggest_for_dir(directory)
        if suggestion is None:
            continue
        if directory == root:
            return suggestion
        relative = directory.relative_to(root)
        cd_target = str(relative).replace("/", "\\") if _IS_WINDOWS else str(relative)
        return f"cd {cd_target} && {suggestion}"
    return None
