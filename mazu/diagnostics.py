import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

MIN_PYTHON = (3, 11)


@dataclass
class CheckResult:
    name: str
    status: str  # "ok", "warn", "fail"
    message: str


def check_python_version(version_info: tuple = sys.version_info) -> CheckResult:
    current = (version_info[0], version_info[1])
    if current >= MIN_PYTHON:
        return CheckResult(
            "Python version", "ok", f"{version_info[0]}.{version_info[1]}.{version_info[2]}"
        )
    return CheckResult(
        "Python version",
        "fail",
        f"{version_info[0]}.{version_info[1]} is below the minimum "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} — checkpoints and some tooling may not work correctly.",
    )


def check_git_available() -> CheckResult:
    if shutil.which("git") is not None:
        return CheckResult("git", "ok", "found on PATH")
    return CheckResult(
        "git", "fail", "not found on PATH — required for checkpoints/rollback (mazu init, mazu run)"
    )


def check_openai_package() -> CheckResult:
    try:
        import openai  # noqa: F401
    except ImportError:
        return CheckResult(
            "openai package",
            "warn",
            "not installed — only needed for openai:*/deepseek:* models. "
            'Install with: pip install "mazu[openai]"',
        )
    return CheckResult("openai package", "ok", "installed")


def check_api_keys() -> list[CheckResult]:
    from mazu.llm.client import _PROVIDERS

    results = []
    any_present = False
    for provider_name, provider in _PROVIDERS.items():
        env_var = provider.api_key_env
        value = os.environ.get(env_var)
        if not value:
            results.append(CheckResult(f"{provider_name} ({env_var})", "warn", "not set"))
            continue
        any_present = True
        if not value.isascii():
            results.append(
                CheckResult(
                    f"{provider_name} ({env_var})",
                    "fail",
                    "contains non-ASCII characters — this can't be a real API key "
                    "(likely a placeholder pasted by mistake)",
                )
            )
        else:
            results.append(CheckResult(f"{provider_name} ({env_var})", "ok", "set"))

    if not any_present:
        results.append(
            CheckResult(
                "API keys",
                "fail",
                "no provider has a key set — mazu chat/run/council will not work until "
                "at least one of ANTHROPIC_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY is set",
            )
        )
    return results


def check_live_api_key(provider_name: str, model: str) -> CheckResult:
    """Makes one minimal real API call to confirm the key actually authenticates --
    NOT run by default (see run_diagnostics' `live` flag) since it costs a fraction
    of a cent and requires network access, unlike every other check here. `model`
    must be a full "provider:model" identifier (e.g. "deepseek:deepseek-chat"), not
    just the bare provider name -- passing a bare provider name here previously sent
    it to run_turn as a model string, which silently defaulted to Anthropic with a
    nonsense model name instead of calling the intended provider at all.
    """
    from mazu.llm.client import run_turn
    from mazu.llm.errors import MazuAPIError, MazuAuthError

    try:
        run_turn(
            messages=[{"role": "user", "content": "hi"}],
            system="Reply with a single word.",
            tools=[],
            model=model,
        )
    except MazuAuthError as e:
        return CheckResult(f"{provider_name} (live)", "fail", f"key rejected: {e}")
    except MazuAPIError as e:
        return CheckResult(f"{provider_name} (live)", "warn", f"couldn't verify (non-auth error): {e}")
    return CheckResult(f"{provider_name} (live)", "ok", "authenticated successfully")


def check_project_git_repo(root: Path) -> CheckResult:
    if (root / ".git").exists():
        return CheckResult("Project git repo", "ok", "initialized")
    return CheckResult(
        "Project git repo", "warn", "not initialized yet — run `mazu init` before `mazu run`"
    )


def check_gitignore(root: Path) -> CheckResult:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return CheckResult(
            ".gitignore", "warn", "no .gitignore yet — `mazu init`/`mazu chat` will create one"
        )
    if ".mazu/" in gitignore.read_text(encoding="utf-8"):
        return CheckResult(".gitignore", "ok", ".mazu/ is excluded")
    return CheckResult(
        ".gitignore",
        "warn",
        ".mazu/ is not excluded — project memory/checkpoints could get committed by accident",
    )


def run_diagnostics(root: Path, live: bool = False) -> list[CheckResult]:
    results = [
        check_python_version(),
        check_git_available(),
        check_openai_package(),
        *check_api_keys(),
        check_project_git_repo(root),
        check_gitignore(root),
    ]
    if live:
        from mazu.llm.client import _PROVIDER_DEFAULT_MODELS, _PROVIDERS

        for provider_name, provider in _PROVIDERS.items():
            if os.environ.get(provider.api_key_env):
                results.append(
                    check_live_api_key(provider_name, _PROVIDER_DEFAULT_MODELS[provider_name])
                )
    return results
