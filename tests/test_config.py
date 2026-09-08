import os

import pytest

from mazu.config import (
    ensure_api_key,
    get_default_model,
    list_config,
    load_config,
    set_config_value,
    unset_config_value,
)


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


# ---------------------------------------------------------------------------
# persistent config: set / list / unset / load
# ---------------------------------------------------------------------------
# Isolation from the real ~/.mazu/config.toml is handled globally by conftest.py's
# autouse _isolate_mazu_config_path fixture -- every test in this file (and every
# other test in the suite) already gets its own tmp_path-backed config file.


def test_list_config_empty_when_no_file():
    assert list_config() == {}


def test_set_and_list_round_trips_a_value():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    assert list_config() == {"default_model": "anthropic:claude-opus-4-8"}


def test_set_unknown_key_raises_value_error():
    with pytest.raises(ValueError, match="Unknown config key"):
        set_config_value("bogus_key", "x")


def test_set_council_lead_accepts_a_well_formed_provider_model_string():
    set_config_value("council_lead", "anthropic:claude-opus-4-8")
    assert list_config()["council_lead"] == "anthropic:claude-opus-4-8"


def test_set_council_lead_accepts_a_bare_model_name():
    # A bare name (no ':') is legitimately valid -- mazu.llm.client._split_model
    # treats it as an Anthropic model, matching MAZU_MODEL=claude-opus-4-8-style
    # config that predates the multi-provider "provider:model" naming.
    set_config_value("council_lead", "claude-opus-4-8")
    assert list_config()["council_lead"] == "claude-opus-4-8"


def test_set_council_lead_rejects_an_empty_provider_half():
    # Real bug found live: the generic Config "Set a value" form accepted
    # council_lead/council_models with zero validation, so a typo like this
    # silently saved and only surfaced later as a confusing, disconnected
    # failure the next time Council actually ran.
    with pytest.raises(ValueError, match="malformed"):
        set_config_value("council_lead", ":claude-opus-4-8")


def test_set_council_lead_rejects_an_empty_model_half():
    with pytest.raises(ValueError, match="malformed"):
        set_config_value("council_lead", "anthropic:")


def test_set_council_lead_rejects_an_empty_value():
    with pytest.raises(ValueError, match="empty"):
        set_config_value("council_lead", "")


def test_set_council_models_validates_every_comma_separated_entry():
    set_config_value("council_models", "anthropic:claude-sonnet-5,openai:gpt-5")
    assert list_config()["council_models"] == "anthropic:claude-sonnet-5,openai:gpt-5"

    with pytest.raises(ValueError, match="malformed"):
        set_config_value("council_models", "anthropic:claude-sonnet-5,openai:")


def test_set_overwrites_existing_value():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    set_config_value("default_model", "deepseek:deepseek-chat")
    assert list_config()["default_model"] == "deepseek:deepseek-chat"


def test_set_preserves_other_keys():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    set_config_value("anthropic_api_key", "sk-ant-123")
    values = list_config()
    assert values["default_model"] == "anthropic:claude-opus-4-8"
    assert values["anthropic_api_key"] == "sk-ant-123"


def test_unset_removes_a_value():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    ok = unset_config_value("default_model")
    assert ok is True
    assert list_config() == {}


def test_unset_missing_key_returns_false_and_changes_nothing():
    set_config_value("default_model", "anthropic:claude-opus-4-8")
    ok = unset_config_value("nonexistent_key")
    assert ok is False
    assert list_config() == {"default_model": "anthropic:claude-opus-4-8"}


def test_concurrent_set_calls_never_corrupt_or_empty_the_config_file():
    # Real bug found via live testing: mazu-web runs Flask with threaded=True, so a
    # burst of near-simultaneous /api/config POSTs used to race on config.toml's
    # non-atomic, unlocked read-modify-write, leaving the file at 0 bytes and
    # silently dropping every previously-set key (API keys, default_model,
    # curator_*). set_config_value now serializes via a file lock and writes
    # atomically (temp file + os.replace) -- this reproduces the concurrency that
    # triggered the corruption and asserts it can no longer happen.
    import concurrent.futures

    from mazu.config import config_path

    def worker(i):
        set_config_value("default_model", f"deepseek:model-{i}")
        set_config_value("curator_model", f"anthropic:model-{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(worker, range(100)))

    # The file must always be well-formed (non-empty, fully parseable) no matter
    # how the 200 individual writes interleaved -- os.replace() is atomic, so a
    # reader landing mid-write always sees either the complete old or complete new
    # content, never a torn/empty file. This does NOT assert the two keys land
    # from the SAME worker's pair of calls: the lock only makes each individual
    # set_config_value() call atomic, not the two separate calls a worker makes
    # back-to-back, so "default_model" and "curator_model" can legitimately end up
    # from different workers depending on scheduling -- that's expected, not a bug.
    assert config_path().stat().st_size > 0
    values = list_config()
    assert set(values) == {"default_model", "curator_model"}
    assert values["default_model"] in {f"deepseek:model-{i}" for i in range(100)}
    assert values["curator_model"] in {f"anthropic:model-{i}" for i in range(100)}


def test_concurrent_set_calls_on_the_same_key_never_lose_an_update():
    # A stronger, unambiguous proof that the lock actually serializes writes (not
    # just that the file stays parseable): a classic read-current-value,
    # increment, write-back counter loses updates under ANY unserialized race, so a
    # final count that exactly matches the number of increments performed is only
    # possible if every read-modify-write ran to completion under the lock without
    # another one interleaved inside it.
    #
    # This calls the private _read_raw_config()/_write_config() directly under one
    # single _config_lock().acquire() per worker, rather than nesting a call to the
    # public set_config_value() (which acquires its own separate lock instance
    # internally) -- filelock.FileLock's reentrancy is per-instance, not global
    # across every FileLock object pointing at the same path from the same thread,
    # so nesting two separate acquisitions like that would deadlock/timeout, not
    # actually test anything. This mirrors exactly what set_config_value() itself
    # does under its own single lock acquisition.
    import concurrent.futures

    from mazu.config import _config_lock, _read_raw_config, _write_config

    # default_model is reused as the counter field simply because it's already a
    # KNOWN_CONFIG_KEYS string value; the "increment" is encoded as an integer
    # suffix (e.g. "deepseek:model-7" means the counter is at 7).
    def worker(_):
        with _config_lock().acquire():
            config = _read_raw_config()
            current = int(config.get("default_model", "deepseek:model-0").rsplit("-", 1)[1])
            config["default_model"] = f"deepseek:model-{current + 1}"
            _write_config(config)

    total_increments = 150
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(worker, range(total_increments)))

    final = list_config()["default_model"]
    assert final == f"deepseek:model-{total_increments}"


def test_set_value_with_special_characters_round_trips_through_toml(tmp_path):
    # Backslashes and quotes must not corrupt the written TOML or break re-parsing.
    tricky = 'sk-"quoted"-and-\\backslash\\-value'
    set_config_value("anthropic_api_key", tricky)
    assert list_config()["anthropic_api_key"] == tricky


def test_get_default_model_returns_none_when_unset():
    assert get_default_model() is None


def test_get_default_model_returns_configured_value():
    set_config_value("default_model", "deepseek:deepseek-chat")
    assert get_default_model() == "deepseek:deepseek-chat"


def test_curator_api_key_is_settable_and_masked():
    from mazu.config import _SECRET_CONFIG_KEYS

    set_config_value("curator_api_key", "sk-curator-secret")
    assert list_config()["curator_api_key"] == "sk-curator-secret"
    assert "curator_api_key" in _SECRET_CONFIG_KEYS


def test_curator_model_and_base_url_and_enabled_are_settable():
    set_config_value("curator_model", "deepseek:deepseek-chat")
    set_config_value("curator_base_url", "http://localhost:1234/v1")
    set_config_value("curator_enabled", "false")
    values = list_config()
    assert values["curator_model"] == "deepseek:deepseek-chat"
    assert values["curator_base_url"] == "http://localhost:1234/v1"
    assert values["curator_enabled"] == "false"


def test_curator_provider_is_not_a_known_key():
    """Deliberately omitted -- curator_model's own 'provider:model' prefix is the
    single source of truth for which provider Curator uses; a separate
    curator_provider key would be a second source of truth that could disagree."""
    with pytest.raises(ValueError, match="Unknown config key"):
        set_config_value("curator_provider", "anthropic")


def test_load_config_never_injects_curator_api_key_into_any_env_var(monkeypatch):
    """The whole point of curator_api_key living outside _PROVIDER_KEY_ENV_VARS:
    load_config()'s env-injection loop (which every real ensure_api_key()/
    default_model() call path runs through) must never touch it or any of the
    main provider env vars because of it."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MAZU_CURATOR_API_KEY", raising=False)
    set_config_value("curator_api_key", "sk-curator-secret")

    load_config()

    assert os.environ.get("ANTHROPIC_API_KEY") is None
    assert os.environ.get("MAZU_CURATOR_API_KEY") is None


def test_load_config_malformed_toml_warns_and_returns_empty(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text("this is not valid toml [[[", encoding="utf-8")
    result = load_config()
    assert result == {}
    assert "malformed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# API key injection into the environment
# ---------------------------------------------------------------------------


def test_load_config_injects_legacy_api_key_as_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    set_config_value("api_key", "sk-legacy-anthropic")
    load_config()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-legacy-anthropic"


def test_load_config_injects_per_provider_keys(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    set_config_value("deepseek_api_key", "sk-deepseek-123")
    load_config()
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-deepseek-123"


def test_load_config_never_overwrites_an_existing_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real-env-value")
    set_config_value("api_key", "sk-config-file-value")
    load_config()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-real-env-value"


def test_load_config_refreshes_a_key_it_previously_injected_itself(monkeypatch):
    # Real bug found via live testing: mazu-web is a long-running process that
    # calls load_config() fresh on every chat turn, but the old "only inject if
    # empty" guard meant the very first turn's key got stuck in os.environ for the
    # rest of the process's life -- rotating the key afterward via the Config UI
    # silently had no effect until the whole server was restarted by hand.
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    set_config_value("deepseek_api_key", "sk-old-key")
    load_config()
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-old-key"

    set_config_value("deepseek_api_key", "sk-new-rotated-key")
    load_config()
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-new-rotated-key"


def test_load_config_still_defers_to_a_real_env_var_set_after_our_own_injection(monkeypatch):
    # Once we've injected a value, a user (or another process) is still free to
    # override it directly -- that override must stick, not get clobbered by our
    # next load_config() refresh.
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    set_config_value("deepseek_api_key", "sk-config-value")
    load_config()
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-config-value"

    os.environ["DEEPSEEK_API_KEY"] = "sk-manually-overridden"
    load_config()
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-manually-overridden"


# ---------------------------------------------------------------------------
# default_model() precedence (client.py, config-file layer)
# ---------------------------------------------------------------------------


def test_default_model_uses_config_when_no_mazu_model_env(monkeypatch):
    from mazu.llm.client import default_model

    monkeypatch.delenv("MAZU_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    set_config_value("default_model", "deepseek:deepseek-chat")

    assert default_model() == "deepseek:deepseek-chat"


def test_default_model_mazu_model_env_wins_over_config(monkeypatch):
    from mazu.llm.client import default_model

    monkeypatch.setenv("MAZU_MODEL", "openai:gpt-5")
    set_config_value("default_model", "deepseek:deepseek-chat")

    assert default_model() == "openai:gpt-5"


def test_default_model_falls_back_to_auto_detect_when_config_unset(monkeypatch):
    from mazu.llm.client import default_model

    monkeypatch.delenv("MAZU_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert default_model() == "deepseek:deepseek-chat"
