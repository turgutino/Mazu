"""`mazu run --protect-tests` adds the "do not edit existing tests" rule to the system prompt,
and leaves the prompt byte-for-byte unchanged when the flag is off.
"""

import subprocess

import pytest

import mazu.agent.autonomous as autonomous_module
from mazu.agent.autonomous import run_autonomous
from mazu.agent.prompts import PROTECT_TESTS_NOTE
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.types import AgentResponse
from mazu.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def _system_prompt_seen(tmp_path, monkeypatch, **kwargs) -> str:
    seen = {}

    def fake_run_turn(messages, system, tools, model=None):
        seen["system"] = system
        return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "done"}], usage={})

    monkeypatch.setattr(autonomous_module, "compact_if_needed", lambda m, model, **k: (m, False))
    monkeypatch.setattr(autonomous_module, "run_turn", fake_run_turn)
    run_autonomous(
        registry=ToolRegistry(), task="do something", session_id="s1",
        checkpoint_manager=CheckpointManager(tmp_path), max_steps=1, **kwargs,
    )
    return seen["system"]


def test_flag_adds_the_rule_to_the_system_prompt(tmp_path, monkeypatch):
    assert PROTECT_TESTS_NOTE in _system_prompt_seen(tmp_path, monkeypatch, protect_tests=True)


def test_flag_off_leaves_the_prompt_alone(tmp_path, monkeypatch):
    assert PROTECT_TESTS_NOTE not in _system_prompt_seen(tmp_path, monkeypatch)
