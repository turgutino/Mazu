"""restore() is a sequence of steps, not one transaction. These tests interrupt it before each step
(by making that step raise) and pin down what is guaranteed:

* an interrupted restore may leave code, memory, and skills out of step with each other, and
* running restore again always repairs it, so the interruption is recoverable.

This simulates a failure at step boundaries. It does not simulate the process being killed in the
middle of writing a file (kill -9, power loss), which is not covered here.
"""

import json
import shutil
import subprocess
import types
from unittest import mock

import pytest

import mazu.checkpoint.manager as mgr
from mazu.checkpoint.manager import CheckpointManager
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager

STEPS = [
    "git reset --hard",
    "git clean -fd",
    "memory.db restore",
    "skills delete",
    "skills copy",
    "read conversation",
    "truncate index",
]
# Steps whose interruption leaves code, memory, and skills in different eras. Pinned so a change in
# restore()'s order is noticed: update this set only after thinking about what the new order means.
MIXED_AFTER_INTERRUPTION = {2, 3, 4, 5}


class Interrupted(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()


def _state_a_then_b(tmp_path):
    """A project checkpointed in state A (cp1), then changed to state B and checkpointed again."""
    p = tmp_path / "proj"
    p.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@t.invalid"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=p, check=True, capture_output=True)
    (p / "a.py").write_text("A\n")
    subprocess.run(["git", "add", "-A"], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=p, check=True, capture_output=True)
    store = MemoryStore(p / ".mazu" / "memory.db")
    store.add(category="fact", title="m1", body="first", source="explicit", session_id="s")
    store.close()
    skills = SkillManager(p)
    skills.save("s1", "d", "def run(args):\n    return 'x'\n")
    manager = CheckpointManager(p)
    cp1 = manager.snapshot([{"role": "user", "content": "A"}], "A")["id"]
    (p / "a.py").write_text("B\n")
    store = MemoryStore(p / ".mazu" / "memory.db")
    store.add(category="fact", title="m2", body="second", source="explicit", session_id="s")
    store.close()
    skills.save("s2", "d", "def run(args):\n    return 'y'\n")
    manager.snapshot([{"role": "user", "content": "B"}], "B")
    return p, manager, cp1


def _observe(p):
    store = MemoryStore(p / ".mazu" / "memory.db")
    memory = sorted(r["title"] for r in store.all_active())
    store.close()
    skills_dir = p / ".mazu" / "skills"
    skills = sorted(d.name for d in skills_dir.iterdir()) if skills_dir.exists() else []
    return (p / "a.py").read_text().strip(), memory, skills


def _eras(state):
    code, memory, skills = state
    return {"A" if code == "A" else "B", "A" if memory == ["m1"] else "B", "A" if skills == ["s1"] else "B"}


def _restore_interrupted_before(manager, checkpoint_id, step_number):
    calls = {"n": 0}

    def tick(real):
        def wrapped(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == step_number:
                raise Interrupted(STEPS[step_number - 1])
            return real(*args, **kwargs)
        return wrapped

    shutil_proxy = types.SimpleNamespace(**{n: getattr(shutil, n) for n in dir(shutil) if not n.startswith("__")})
    shutil_proxy.rmtree = tick(shutil.rmtree)
    shutil_proxy.copytree = tick(shutil.copytree)
    json_proxy = types.SimpleNamespace(**{n: getattr(json, n) for n in dir(json) if not n.startswith("__")})
    json_proxy.loads = tick(json.loads)
    index_cls = type(manager.index)
    with mock.patch.object(mgr, "_git", tick(mgr._git)), \
            mock.patch.object(mgr, "_backup_sqlite", tick(mgr._backup_sqlite)), \
            mock.patch.object(mgr, "shutil", shutil_proxy), mock.patch.object(mgr, "json", json_proxy), \
            mock.patch.object(index_cls, "truncate_after", tick(index_cls.truncate_after)):
        with pytest.raises(Interrupted):
            manager.restore(checkpoint_id)


@pytest.mark.parametrize("step_number", range(1, len(STEPS) + 1), ids=STEPS)
def test_interrupted_restore_is_recoverable_by_running_it_again(tmp_path, step_number):
    p, manager, cp1 = _state_a_then_b(tmp_path)

    _restore_interrupted_before(manager, cp1, step_number)
    interrupted_state = _observe(p)
    assert (len(_eras(interrupted_state)) > 1) == (step_number in MIXED_AFTER_INTERRUPTION), (
        f"interrupted before '{STEPS[step_number - 1]}': {interrupted_state}"
    )

    manager.restore(cp1)  # simply run it again

    assert _observe(p) == ("A", ["m1"], ["s1"])
    assert [e["id"] for e in manager.list_checkpoints()] == [cp1]
