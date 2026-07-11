import pytest

from mazu.config import ensure_api_key


def test_missing_key_shows_set_syntax_on_windows(monkeypatch):
    monkeypatch.setattr("mazu.config.platform.system", lambda: "Windows")
    # Re-import to pick up the module-level _SET_ENV_EXAMPLE computed from
    # platform.system() at import time -- patch it directly instead, matching
    # what the code actually reads.
    monkeypatch.setattr("mazu.config._SET_ENV_EXAMPLE", "set {var}=...")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("MAZU_MODEL", "deepseek:deepseek-chat")

    with pytest.raises(SystemExit) as exc_info:
        ensure_api_key()

    message = str(exc_info.value)
    assert "set DEEPSEEK_API_KEY=..." in message
    assert "export" not in message


def test_missing_key_shows_export_syntax_on_posix(monkeypatch):
    monkeypatch.setattr("mazu.config._SET_ENV_EXAMPLE", "export {var}=...")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("MAZU_MODEL", "deepseek:deepseek-chat")

    with pytest.raises(SystemExit) as exc_info:
        ensure_api_key()

    message = str(exc_info.value)
    assert "export DEEPSEEK_API_KEY=..." in message


def test_key_present_does_not_raise(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setenv("MAZU_MODEL", "deepseek:deepseek-chat")
    ensure_api_key()  # should not raise
