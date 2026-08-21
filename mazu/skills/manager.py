import json
import os
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
        """Writes/overwrites the skill's code, but MERGES meta.json over any
        existing entry rather than replacing it wholesale -- previously this reset
        usage_count/created_at to 0/now on every save, which would destroy a
        skill's own usage evidence the moment anything (a human, or Curator)
        rewrote its code to fix a bug. New/never-seen fields (archived,
        success_count, curator_revision, ...) default sensibly for a first save.
        """
        if not NAME_RE.match(name):
            raise ValueError(
                "name must be a valid identifier: letters, digits, underscore, not starting with a digit"
            )
        if "def run(" not in code:
            raise ValueError("code must define a function `def run(args: dict) -> str:`")
        if '__name__ == "__main__"' in code or "__name__ == '__main__'" in code:
            # Real bug caught via live testing: `code` must be ONLY the `def run(...)`
            # function body -- SKILL_TEMPLATE below already supplies the
            # `if __name__ == "__main__":` entry point and imports. A caller (human
            # or an LLM tool call, e.g. Curator's write_skill) that pastes a
            # complete standalone script -- including its own copy of that guard --
            # would otherwise get it wrapped a SECOND time by SKILL_TEMPLATE.format,
            # producing a file with the guard duplicated. Both copies execute when
            # the file runs as __main__: the first reads stdin and returns the
            # correct answer, but the second then re-reads (now-exhausted) stdin,
            # gets `{}`, and crashes -- so the skill looks fixed (right code, right
            # reasoning) but is actually broken on every single invocation. Failing
            # loudly here beats silently shipping a skill that's broken this way.
            raise ValueError(
                "code must be ONLY the `def run(args: dict) -> str:` function body -- "
                "do not include imports, an `if __name__ == \"__main__\":` block, or any "
                "other script boilerplate; that wrapper is added automatically."
            )
        skill_dir = self._dir(name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "skill.py").write_text(SKILL_TEMPLATE.format(code=code), encoding="utf-8")

        existing = self.get_meta(name) or {}
        meta = {
            "name": name,
            "description": description,
            "tags": tags,
            "created_at": existing.get("created_at", datetime.now(timezone.utc).isoformat()),
            "usage_count": existing.get("usage_count", 0),
            "last_used_at": existing.get("last_used_at"),
            "archived": existing.get("archived", False),
            "archived_at": existing.get("archived_at"),
            "archived_reason": existing.get("archived_reason"),
            "superseded_by": existing.get("superseded_by"),
            "success_count": existing.get("success_count", 0),
            "failure_count": existing.get("failure_count", 0),
            "last_outcome": existing.get("last_outcome"),
            "curator_revision": existing.get("curator_revision", 0),
            "curator_last_edited_at": existing.get("curator_last_edited_at"),
        }
        self._write_meta(name, meta)

    def get_meta(self, name: str) -> dict | None:
        meta_path = self._dir(name) / "meta.json"
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_meta(self, name: str, meta: dict) -> None:
        (self._dir(name) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def read_code(self, name: str) -> str | None:
        skill_path = self._dir(name) / "skill.py"
        if not skill_path.exists():
            return None
        return skill_path.read_text(encoding="utf-8")

    def list(self, include_archived: bool = False) -> list[dict]:
        if not self.skills_dir.exists():
            return []
        metas = []
        for meta_file in sorted(self.skills_dir.glob("*/meta.json")):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[skills] warning: could not read {meta_file}: {e}")
                continue
            if meta.get("archived") and not include_archived:
                continue
            metas.append(meta)
        return metas

    def exists(self, name: str) -> bool:
        return (self._dir(name) / "skill.py").exists()

    def archive(self, name: str, reason: str | None = None) -> bool:
        """Reversible retirement -- mirrors MemoryStore.archive()'s semantics.
        The skill's code and directory stay on disk; it just stops appearing in
        list()/build_context_block()'s default (active-only) view."""
        meta = self.get_meta(name)
        if meta is None:
            return False
        meta["archived"] = True
        meta["archived_at"] = datetime.now(timezone.utc).isoformat()
        meta["archived_reason"] = reason
        self._write_meta(name, meta)
        return True

    def unarchive(self, name: str) -> bool:
        meta = self.get_meta(name)
        if meta is None:
            return False
        meta["archived"] = False
        meta["archived_at"] = None
        meta["archived_reason"] = None
        self._write_meta(name, meta)
        return True

    def update_meta(self, name: str, **fields) -> bool:
        meta = self.get_meta(name)
        if meta is None:
            return False
        meta.update(fields)
        self._write_meta(name, meta)
        return True

    def supersede(self, old_name: str, new_name: str) -> bool:
        """Mirrors MemoryStore.supersede() -- old_name is archived (not deleted)
        and tagged with which skill replaced it, for the same audit-trail reason."""
        old_meta = self.get_meta(old_name)
        if old_meta is None or self.get_meta(new_name) is None:
            return False
        old_meta["archived"] = True
        old_meta["archived_at"] = datetime.now(timezone.utc).isoformat()
        old_meta["archived_reason"] = f"superseded by {new_name}"
        old_meta["superseded_by"] = new_name
        self._write_meta(old_name, old_meta)
        return True

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
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                # Same fix as run_shell (mazu/tools/shell.py): on Windows, a skill
                # script's own stdout defaults to the console's legacy codepage, not
                # UTF-8, so printing non-ASCII text (an emoji, a Turkish/Azerbaijani
                # letter) crashes the skill subprocess itself with its own
                # UnicodeEncodeError -- a real, observed failure during live
                # testing, not a hypothetical one.
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
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
