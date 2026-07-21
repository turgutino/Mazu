from pathlib import Path

from mazu.tools.shell import (
    denylist_reason,
    is_allowed_by_shell_allowlist,
    is_denied_shell_command,
    long_running_server_reason,
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


# ---------------------------------------------------------------------------
# long_running_server_reason -- real bug found via live testing: python app.py
# (a Flask dev server) blocked run_shell until the user manually interrupted the
# whole mazu process, since a server never exits on its own.
# ---------------------------------------------------------------------------


def test_long_running_server_reason_none_for_ordinary_commands(tmp_path: Path):
    assert long_running_server_reason("git status", tmp_path) is None
    assert long_running_server_reason("npm test", tmp_path) is None
    assert long_running_server_reason("npm run build", tmp_path) is None
    assert long_running_server_reason("pytest -q", tmp_path) is None


def test_long_running_server_reason_matches_command_line_patterns(tmp_path: Path):
    assert long_running_server_reason("flask run", tmp_path) is not None
    assert long_running_server_reason("npm run dev", tmp_path) is not None
    assert long_running_server_reason("npm start", tmp_path) is not None
    assert long_running_server_reason("yarn dev", tmp_path) is not None
    assert long_running_server_reason("vite", tmp_path) is not None
    assert long_running_server_reason("next dev", tmp_path) is not None
    assert long_running_server_reason("uvicorn app:app", tmp_path) is not None
    assert long_running_server_reason("gunicorn app:app", tmp_path) is not None
    assert long_running_server_reason("python -m http.server 8000", tmp_path) is not None
    assert long_running_server_reason("python manage.py runserver", tmp_path) is not None
    assert long_running_server_reason("rails server", tmp_path) is not None


def test_long_running_server_reason_vite_build_is_not_flagged(tmp_path: Path):
    # "vite build" finishes on its own -- only bare "vite" (implicitly the dev
    # server) should be flagged.
    assert long_running_server_reason("vite build", tmp_path) is None


def test_long_running_server_reason_inspects_bare_python_script_source(tmp_path: Path):
    """The real bug: `python app.py` gives no hint on the command line itself that
    app.py starts a Flask server -- that only shows up in the file's own source.
    """
    (tmp_path / "app.py").write_text(
        "from flask import Flask\napp = Flask(__name__)\napp.run(debug=True)\n",
        encoding="utf-8",
    )
    reason = long_running_server_reason("python app.py", tmp_path)
    assert reason is not None
    assert "app.py" in reason


def test_long_running_server_reason_bare_python_script_without_server_signature(tmp_path: Path):
    (tmp_path / "one.py").write_text("print('hello')\n", encoding="utf-8")
    assert long_running_server_reason("python one.py", tmp_path) is None


def test_long_running_server_reason_ignores_compound_commands(tmp_path: Path):
    # The bare-script source check only applies to a command that is EXACTLY
    # `python <file>.py` -- a compound command (&&, extra args) isn't parsed for a
    # target file, so it falls through unflagged (a known, accepted false negative).
    (tmp_path / "app.py").write_text("app.run(debug=True)\n", encoding="utf-8")
    assert long_running_server_reason("cd sub && python app.py", tmp_path) is None
    assert long_running_server_reason("python app.py --port 8080", tmp_path) is None


def test_long_running_server_reason_missing_script_file_is_safe(tmp_path: Path):
    assert long_running_server_reason("python does_not_exist.py", tmp_path) is None


def test_shell_tool_blocks_server_command_without_running_it(tmp_path: Path):
    (tmp_path / "app.py").write_text(
        "from flask import Flask\napp = Flask(__name__)\napp.run(debug=True)\n",
        encoding="utf-8",
    )
    tool = make_shell_tool(tmp_path, dry_run=False)
    result = tool.handler({"command": "python app.py"})
    assert result.is_error
    assert "Not run" in result.content
    assert "own terminal" in result.content


def test_shell_tool_handles_non_ascii_subprocess_output_without_crashing(tmp_path: Path):
    """Regression test for a real bug found via live testing (twice, independently:
    an emoji in a generated print statement, then a Turkish/Azerbaijani letter): on
    Windows, a spawned Python subprocess's own stdout defaults to the console's
    legacy codepage, not UTF-8, so printing non-ASCII text used to crash the
    *subprocess* itself with its own UnicodeEncodeError. PYTHONIOENCODING is now set
    for every run_shell subprocess to prevent this at the source.
    """
    tool = make_shell_tool(tmp_path, dry_run=False)
    # chr(0x1F600) is an emoji (grinning face); chr(305) is Turkish/Azerbaijani
    # dotless lowercase "ı" -- both previously observed to crash a subprocess.
    result = tool.handler(
        {"command": f'python -c "print(chr({0x1F600})); print(chr({305}))"'}
    )
    assert not result.is_error
    assert chr(0x1F600) in result.content
    assert chr(305) in result.content
