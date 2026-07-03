import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

SKILL_TEMPLATE = """\
import json
import sys

{code}

if __name__ == "__main__":
    raw = sys.stdin.read()
    args = json.loads(raw) if raw.strip() else {{}}
    print(run(args))
"""


class SkillManager:
    """A self-growing local library of reusable Python functions the agent has written
    for itself. Running a saved skill is a local subprocess call, not an API call — the
    whole point is to let repeated tasks skip Claude entirely once solved once.
    """

    def __init__(self, root: Path):
        self.root = root
        self.skills_dir = root / ".mazu" / "skills"

    def _dir(self, name: str) -> Path:
        return self.skills_dir / name

    def save(self, name: str, description: str, code: str, tags: str = "") -> None:
        if not NAME_RE.match(name):
            raise ValueError(
                "name must be a valid identifier: letters, digits, underscore, not starting with a digit"
            )
        if "def run(" not in code:
            raise ValueError("code must define a function `def run(args: dict) -> str:`")
        skill_dir = self._dir(name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "skill.py").write_text(SKILL_TEMPLATE.format(code=code), encoding="utf-8")
        meta = {
            "name": name,
            "description": description,
            "tags": tags,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "usage_count": 0,
            "last_used_at": None,
        }
        (skill_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def list(self) -> list[dict]:
        if not self.skills_dir.exists():
            return []
        metas = []
        for meta_file in sorted(self.skills_dir.glob("*/meta.json")):
            try:
                metas.append(json.loads(meta_file.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"[skills] warning: could not read {meta_file}: {e}")
                continue
        return metas

    def exists(self, name: str) -> bool:
        return (self._dir(name) / "skill.py").exists()

    def run(self, name: str, args: dict, timeout: int = 60) -> tuple[str, bool]:
        skill_path = self._dir(name) / "skill.py"
        if not skill_path.exists():
            return f"No skill named '{name}'", True
        try:
            # Args go via stdin, not argv -- a command-line argument has a ~32KB
            # limit on Windows, which large skill args could exceed; stdin has no
            # such practical limit.
            proc = subprocess.run(
                [sys.executable, str(skill_path)],
                input=json.dumps(args),
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Skill '{name}' timed out after {timeout}s", True
        self._bump_usage(name)
        output = proc.stdout
        if proc.stderr:
            output += "\n--- stderr ---\n" + proc.stderr
        return output, proc.returncode != 0

    def _bump_usage(self, name: str) -> None:
        meta_path = self._dir(name) / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["usage_count"] = meta.get("usage_count", 0) + 1
            meta["last_used_at"] = datetime.now(timezone.utc).isoformat()
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    def delete(self, name: str) -> bool:
        skill_dir = self._dir(name)
        if not skill_dir.exists():
            return False
        shutil.rmtree(skill_dir)
        return True

    def build_context_block(self) -> str:
        metas = self.list()
        if not metas:
            return ""
        lines = [
            "## Available Skills (auto-loaded)",
            "These are reusable local functions saved from past sessions on this project. "
            "Prefer calling run_skill over re-deriving the logic when one of these already "
            "matches the current need.",
            "",
        ]
        for m in metas:
            lines.append(
                f"- {m['name']}: {m['description']} "
                f"(used {m.get('usage_count', 0)}x, tags: {m.get('tags') or '-'})"
            )
        return "\n".join(lines) + "\n"
