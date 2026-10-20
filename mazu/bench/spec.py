"""Benchmark suite and configuration files for `mazu bench`.

A suite is a list of tasks, each pointing at a local git repository, a task prompt,
and a *verify command* that is run after the agent finishes and decides pass/fail
independently of anything the agent claims (the research behind this: an agent saying
"done" and its own tests are weak correctness signals, an external check is not).
A config is a named set of CLI flags / environment overrides for `mazu run`, so the
same suite can be replayed under different settings and compared.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Task:
    id: str
    repo: str
    prompt: str
    verify_command: str
    max_steps: int = 15
    ref: str | None = None


@dataclass(frozen=True)
class Config:
    name: str
    model: str | None = None
    extra_args: tuple[str, ...] = ()
    env: dict = field(default_factory=dict)


def _require(item: dict, key: str, kind: str):
    if key not in item or item[key] in (None, ""):
        raise ValueError(f"{kind} entry is missing required field '{key}': {item!r}")
    return item[key]


def load_suite(path: Path) -> list[Task]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("A suite file must contain a non-empty JSON list of tasks.")
    tasks = []
    seen = set()
    for item in data:
        task_id = _require(item, "id", "task")
        if task_id in seen:
            raise ValueError(f"Duplicate task id in suite: {task_id!r}")
        seen.add(task_id)
        tasks.append(
            Task(
                id=task_id,
                repo=_require(item, "repo", "task"),
                prompt=_require(item, "prompt", "task"),
                verify_command=_require(item, "verify_command", "task"),
                max_steps=int(item.get("max_steps", 15)),
                ref=item.get("ref"),
            )
        )
    return tasks


def load_configs(path: Path) -> list[Config]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("A configs file must contain a non-empty JSON list of configs.")
    configs = []
    seen = set()
    for item in data:
        name = _require(item, "name", "config")
        if name in seen:
            raise ValueError(f"Duplicate config name: {name!r}")
        seen.add(name)
        configs.append(
            Config(
                name=name,
                model=item.get("model"),
                extra_args=tuple(item.get("extra_args", [])),
                env={str(k): str(v) for k, v in item.get("env", {}).items()},
            )
        )
    return configs
