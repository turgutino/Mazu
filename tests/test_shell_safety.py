from pathlib import Path

from mazu.tools.shell import (
    denylist_reason,
    is_allowed_by_shell_allowlist,
    is_denied_shell_command,
    make_shell_tool,
)


# ---------------------------------------------------------------------------
# denylist_reason / is_denied_shell_command
# ---------------------------------------------------------------------------


def test_denylist_reason_none_for_safe_command():
    assert denylist_reason("git status") is None
    assert is_denied_shell_command("git status") is False


def test_denylist_reason_returns_human_readable_text_for_sudo():
    reason = denylist_reason("sudo rm somefile")
    assert reason is not None
    assert "sudo" in reason.lower()
    assert is_denied_shell_command("sudo rm somefile") is True


def test_denylist_reason_matches_rm_rf_root():
    reason = denylist_reason("rm -rf /")
    assert reason is not None
    assert "root" in reason.lower()


def test_denylist_reason_matches_force_push():
    reason = denylist_reason("git push origin main --force")
    assert reason is not None
    assert "force" in reason.lower()


def test_denylist_reason_matches_ssh_dir():
    reason = denylist_reason("cat ~/.ssh/id_rsa")
    assert reason is not None


def test_denylist_reason_matches_format_drive():
    reason = denylist_reason("format c:")
    assert reason is not None


# ---------------------------------------------------------------------------
# is_allowed_by_shell_allowlist
# ---------------------------------------------------------------------------


def test_no_allowlist_configured_allows_everything():
    assert is_allowed_by_shell_allowlist("anything at all", None) is True
    assert is_allowed_by_shell_allowlist("anything at all", []) is True


def test_allowlist_permits_matching_program():
    assert is_allowed_by_shell_allowlist("git status", ["git"]) is True


def test_allowlist_blocks_non_matching_program():
    assert is_allowed_by_shell_allowlist("rm -rf build", ["git", "npm"]) is False


def test_allowlist_requires_whole_word_match_not_substring():
    # "gitx" must not be treated as an allowed "git" invocation just because it
    # starts with the same letters.
    assert is_allowed_by_shell_allowlist("gitx status", ["git"]) is False


def test_allowlist_does_not_match_program_name_appearing_later_in_command():
    # "echo git" mentions "git" but doesn't start with it -- must not be allowed
    # just because the token appears somewhere in the command.
    assert is_allowed_by_shell_allowlist("echo git", ["git"]) is False


def test_allowlist_ignores_leading_whitespace():
    assert is_allowed_by_shell_allowlist("   git status", ["git"]) is True


def test_allowlist_matches_any_entry_in_the_list():
    assert is_allowed_by_shell_allowlist("pytest -q", ["git", "npm", "pytest"]) is True


# ---------------------------------------------------------------------------
# make_shell_tool dry-run mode
# ---------------------------------------------------------------------------


def test_dry_run_shell_tool_does_not_execute(tmp_path: Path):
    marker = tmp_path / "should_not_exist.txt"
    tool = make_shell_tool(tmp_path, dry_run=True)
    result = tool.handler({"command": f"echo hi > {marker.name}"})

    assert not result.is_error
    assert "[dry-run]" in result.content
    assert "Would run" in result.content
    assert not marker.exists()


def test_dry_run_shell_tool_reports_the_command(tmp_path: Path):
    tool = make_shell_tool(tmp_path, dry_run=True)
    result = tool.handler({"command": "npm install"})
    assert "npm install" in result.content


def test_non_dry_run_shell_tool_still_executes_for_real(tmp_path: Path):
    tool = make_shell_tool(tmp_path, dry_run=False)
    result = tool.handler({"command": "echo hello"})
    assert not result.is_error
    assert "hello" in result.content
