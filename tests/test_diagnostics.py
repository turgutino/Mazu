import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mazu.diagnostics import (
    apply_fixes,
    check_api_keys,
    check_git_available,
    check_gitignore,
    check_live_api_key,
    check_openai_package,
    check_project_git_repo,
    check_python_version,
    ensure_gitignore,
    run_diagnostics,
)
from mazu.llm.errors import MazuAuthError, MazuTransientError


# ---------------------------------------------------------------------------
# check_python_version
# ---------------------------------------------------------------------------


def test_python_version_ok_when_above_minimum():
    result = check_python_version(version_info=(3, 12, 1, "final", 0))
    assert result.status == "ok"


def test_python_version_fails_when_below_minimum():
    result = check_python_version(version_info=(3, 9, 0, "final", 0))
    assert result.status == "fail"


def test_python_version_ok_at_exact_minimum():
    result = check_python_version(version_info=(3, 11, 0, "final", 0))
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# check_git_available
# ---------------------------------------------------------------------------


def test_git_available_ok(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "C:/git/git.exe")
    assert check_git_available().status == "ok"


def test_git_missing_fails(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert check_git_available().status == "fail"


# ---------------------------------------------------------------------------
# check_openai_package
# ---------------------------------------------------------------------------


def test_openai_package_installed_ok():
    # Actually installed in the dev/test environment (mazu[openai] extra).
    result = check_openai_package()
    assert result.status == "ok"


def test_openai_package_missing_warns(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    result = check_openai_package()
    assert result.status == "warn"


# ---------------------------------------------------------------------------
# check_api_keys
# ---------------------------------------------------------------------------


def test_no_keys_set_reports_fail(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    results = check_api_keys()

    statuses = {r.status for r in results}
    assert "fail" in statuses  # the overall "no key at all" summary row
    assert all(r.status == "warn" for r in results if r.name != "API keys")


def test_one_key_set_reports_ok_for_that_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-key")

    results = check_api_keys()

    deepseek_result = next(r for r in results if r.name.startswith("deepseek"))
    assert deepseek_result.status == "ok"
    assert not any(r.status == "fail" for r in results)  # no "nothing at all" summary row


def test_non_ascii_key_reports_fail(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sənin-key-in")

    results = check_api_keys()

    deepseek_result = next(r for r in results if r.name.startswith("deepseek"))
    assert deepseek_result.status == "fail"
    assert "non-ASCII" in deepseek_result.message


# ---------------------------------------------------------------------------
# check_project_git_repo / check_gitignore
# ---------------------------------------------------------------------------


def test_project_git_repo_ok_when_initialized(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert check_project_git_repo(tmp_path).status == "ok"


def test_project_git_repo_warns_when_missing(tmp_path: Path):
    assert check_project_git_repo(tmp_path).status == "warn"


def test_gitignore_warns_when_absent(tmp_path: Path):
    assert check_gitignore(tmp_path).status == "warn"


def test_gitignore_ok_when_mazu_dir_excluded(tmp_path: Path):
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    assert check_gitignore(tmp_path).status == "ok"


def test_gitignore_warns_when_mazu_dir_not_excluded(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    assert check_gitignore(tmp_path).status == "warn"


# ---------------------------------------------------------------------------
# check_live_api_key (mocked -- the real network verification happens
# separately, live, with real keys, not as part of the automated suite)
# ---------------------------------------------------------------------------


def test_live_check_auth_error_reports_fail():
    with patch("mazu.llm.client.run_turn", side_effect=MazuAuthError("bad key")):
        result = check_live_api_key("anthropic", "anthropic:claude-sonnet-5")
    assert result.status == "fail"
    assert "rejected" in result.message


def test_live_check_success_reports_ok():
    from mazu.llm.types import AgentResponse

    fake_response = AgentResponse(stop_reason="end_turn", content=[], usage={})
    with patch("mazu.llm.client.run_turn", return_value=fake_response):
        result = check_live_api_key("anthropic", "anthropic:claude-sonnet-5")
    assert result.status == "ok"


def test_live_check_non_auth_error_reports_warn_not_fail():
    with patch("mazu.llm.client.run_turn", side_effect=MazuTransientError("timeout")):
        result = check_live_api_key("anthropic", "anthropic:claude-sonnet-5")
    assert result.status == "warn"


def test_live_check_uses_correct_model_for_provider():
    """Regression test for a real bug: run_diagnostics used to pass the bare
    provider name (e.g. "deepseek") as the model string, which run_turn's
    _split_model() would silently interpret as an Anthropic model named
    "deepseek" instead of actually calling DeepSeek at all.
    """
    captured = {}

    def _fake_run_turn(messages, system, tools, model=None):
        captured["model"] = model
        from mazu.llm.types import AgentResponse

        return AgentResponse(stop_reason="end_turn", content=[], usage={})

    with patch("mazu.llm.client.run_turn", side_effect=_fake_run_turn):
        check_live_api_key("deepseek", "deepseek:deepseek-chat")

    assert captured["model"] == "deepseek:deepseek-chat"


# ---------------------------------------------------------------------------
# run_diagnostics
# ---------------------------------------------------------------------------


def test_run_diagnostics_skips_live_checks_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    with patch("mazu.diagnostics.check_live_api_key") as mock_live:
        run_diagnostics(tmp_path, live=False)
    mock_live.assert_not_called()


def test_run_diagnostics_runs_live_check_only_for_configured_providers(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    with patch("mazu.diagnostics.check_live_api_key") as mock_live:
        mock_live.return_value = None
        run_diagnostics(tmp_path, live=True)

    called_providers = [call.args[0] for call in mock_live.call_args_list]
    assert called_providers == ["deepseek"]


# ---------------------------------------------------------------------------
# ensure_gitignore
# ---------------------------------------------------------------------------


def test_ensure_gitignore_creates_file_when_absent(tmp_path: Path):
    ensure_gitignore(tmp_path)
    assert ".mazu/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_ensure_gitignore_appends_when_file_exists_without_entry(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    ensure_gitignore(tmp_path)
    content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in content
    assert ".mazu/" in content


def test_ensure_gitignore_is_a_noop_when_already_present(tmp_path: Path):
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    before = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    ensure_gitignore(tmp_path)
    after = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert before == after


# ---------------------------------------------------------------------------
# apply_fixes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def test_apply_fixes_creates_missing_gitignore(tmp_path: Path):
    fixed = apply_fixes(tmp_path)
    assert any("gitignore" in f.lower() for f in fixed)
    assert ".mazu/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_apply_fixes_initializes_git_repo(tmp_path: Path):
    assert not (tmp_path / ".git").exists()
    fixed = apply_fixes(tmp_path)
    assert any("git repository" in f.lower() for f in fixed)
    assert (tmp_path / ".git").exists()


def test_apply_fixes_reports_nothing_when_already_correct(tmp_path: Path):
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

    fixed = apply_fixes(tmp_path)
    assert fixed == []


def test_apply_fixes_only_fixes_gitignore_when_git_already_initialized(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

    fixed = apply_fixes(tmp_path)
    assert len(fixed) == 1
    assert "gitignore" in fixed[0].lower()


def test_apply_fixes_is_idempotent(tmp_path: Path):
    apply_fixes(tmp_path)
    second_pass = apply_fixes(tmp_path)
    assert second_pass == []
