import os
import platform
import tempfile
import tomllib
from pathlib import Path

from mazu.checkpoint.lock import ReentrantFileLock


def _config_lock() -> ReentrantFileLock:
    # Re-evaluated on every call (never a frozen module-level lock object), for the
    # same reason config_path() itself is a function: it must track a HOME/
    # USERPROFILE override made mid-session (tests, or `mazu-web` pointed at a
    # different project). Keyed on config_path() itself, so `mazu` and `mazu-web`
    # processes sharing the same ~/.mazu/config.toml serialize against each other too,
    # not just threads within one process.
    return ReentrantFileLock(config_path())


def config_path() -> Path:
    # A function, not a frozen module-level constant, deliberately -- Path.home()
    # must be re-evaluated on every call so it picks up a HOME/USERPROFILE override
    # (tests, or a user who changes it mid-session) the same way every other
    # ~/.mazu/*-path helper in this codebase already does (see cli.py's
    # _usage_db_path()/_action_log_db_path()). A frozen constant computed once at
    # import time would silently keep pointing at whatever HOME was at import time.
    return Path.home() / ".mazu" / "config.toml"

# `export VAR=...` is bash/zsh syntax and does nothing useful if pasted into
# Windows cmd.exe (it either errors or silently sets a literal variable named
# "export" nobody reads) -- show the syntax that actually works in the user's shell.
_SET_ENV_EXAMPLE = "set {var}=..." if platform.system() == "Windows" else "export {var}=..."

# Per-provider API key config keys -> the env var each one is injected as.
# `api_key` (no prefix) is kept as a legacy alias for `anthropic_api_key`, so a
# config.toml written before per-provider keys existed keeps working unchanged.
_PROVIDER_KEY_ENV_VARS = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
}
# Curator's own model/key/enablement -- deliberately NOT part of _PROVIDER_KEY_ENV_VARS
# and never touched by load_config()'s env-injection loop below. That loop's whole
# purpose is auto-injecting the *shared* provider keys the main model resolution
# consults (ensure_api_key()/default_model()); curator_api_key must never be able to
# leak into that shared environ pool, and the main key must never be picked up by
# Curator -- see mazu/curator/config.py, which reads these independently.
# curator_provider is deliberately omitted: curator_model is "provider:model" and the
# provider is derived from its prefix (same as every other model string in Mazu) --
# a separate curator_provider key would be a second source of truth that can silently
# disagree with curator_model's own prefix.
_CURATOR_CONFIG_KEYS = {"curator_api_key", "curator_model", "curator_base_url", "curator_enabled"}
# A persisted default roster for `mazu council` -- previously pure hardcoded CLI
# defaults (DEFAULT_COUNCIL_MODELS/DEFAULT_COUNCIL_LEAD in cli.py), which stay as
# the fallback when these are unset. council_models is a comma-joined model list,
# matching how --models is already parsed on the command line.
_COUNCIL_CONFIG_KEYS = {"council_models", "council_lead"}

# Tracks, per env var, the last value *this process* injected via load_config()'s
# env-injection loop below. Real bug found via live testing: mazu-web is a
# long-running process that calls load_config() fresh on every chat/task turn
# (see mazu_web/chat_session.py), but the injection loop only ever wrote a key
# into os.environ if that var was still empty -- so the very first turn's config
# value got stuck in os.environ for the rest of the process's life. Rotating the
# key afterward via the Config UI updated config.toml on disk, but the running
# server kept sending the stale first-loaded key to the provider until it was
# restarted by hand. Comparing against the last value *we* injected (rather than
# just "is the env var non-empty") lets a config.toml change flow through on the
# very next load_config() call, while a value the user set directly in their own
# shell -- never recorded here -- is still never touched, preserving the existing
# "real env vars always win" guarantee the tests below depend on.
_last_injected: dict[str, str] = {}
KNOWN_CONFIG_KEYS = {
    "default_model", "api_key", "local_base_url", "router_suggestions", "reasoning_effort",
    *_PROVIDER_KEY_ENV_VARS, *_CURATOR_CONFIG_KEYS, *_COUNCIL_CONFIG_KEYS,
}
# Keys whose stored value is a secret -- `mazu config list` masks these, never
# printing the real value. local_base_url is deliberately excluded: it's a URL, not
# a secret, and showing it in plain text is what makes `mazu config list` useful for
# confirming which local server Mazu is actually pointed at.
_SECRET_CONFIG_KEYS = {"api_key", *_PROVIDER_KEY_ENV_VARS, "curator_api_key"}

_LOCAL_BASE_URL_DEFAULT = "http://localhost:1234/v1"  # LM Studio's default port


def router_suggestions_enabled() -> bool:
    """Default True. `mazu config set router_suggestions false` silences the
    passive "based on past explorations, consider --model X" print in `mazu run`/
    `mazu chat` -- useful for scripting/CI, where an unexpected extra printed line
    could pollute captured output. Stored as a plain string ("true"/"false") like
    every other config value; not a secret, not added to _SECRET_CONFIG_KEYS.
    """
    return _read_raw_config().get("router_suggestions", "true").lower() != "false"


def _read_raw_config() -> dict:
    path = config_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"[config] warning: {path} is malformed ({e}); ignoring it.")
        return {}


def _inject_env_var(env_var: str, value: str) -> None:
    """Sets `env_var` from config unless something other than our own previous
    injection currently occupies it -- see `_last_injected`'s comment for why this
    is not simply "if not already set"."""
    current = os.environ.get(env_var)
    if current is not None and current != _last_injected.get(env_var):
        return  # a real external env var (or one we've never touched) -- hands off
    os.environ[env_var] = value
    _last_injected[env_var] = value


def load_config() -> dict:
    """Load ~/.mazu/config.toml if present and inject any configured API keys into
    the environment -- env vars set directly always take priority and are never
    overwritten. `api_key` is a legacy alias for `anthropic_api_key`. A value
    previously injected by this same function is refreshed on every call, so a key
    rotated via `mazu config set` / the mazu-web Config page takes effect on the
    next call (the next chat/task turn) without requiring a process restart.
    """
    config = _read_raw_config()

    legacy_anthropic_key = config.get("api_key")
    if legacy_anthropic_key:
        _inject_env_var("ANTHROPIC_API_KEY", legacy_anthropic_key)

    for config_key, env_var in _PROVIDER_KEY_ENV_VARS.items():
        value = config.get(config_key)
        if value:
            _inject_env_var(env_var, value)

    return config


def get_default_model() -> str | None:
    """The `default_model` set via `mazu config set default_model provider:model`, or
    None if unset. Consulted by client.default_model() after the MAZU_MODEL env var
    (which always wins, matching how config-file API keys never override env vars
    either) and before auto-detecting from whichever provider's key is present.
    """
    return _read_raw_config().get("default_model")


def get_reasoning_effort() -> str | None:
    """The `reasoning_effort` set via `mazu config set reasoning_effort low|medium|high`
    (or the mazu-web Config/Chat pages), or None if unset. Consulted by
    client.default_effort() after the MAZU_REASONING_EFFORT env var. Not validated
    against {"low","medium","high"} here -- an unsupported value is left to surface
    as a normal provider-side error, same as an invalid model string.
    """
    return _read_raw_config().get("reasoning_effort")


def list_config() -> dict:
    """Raw config exactly as stored on disk -- callers displaying this to a user
    should mask secret values themselves (see _SECRET_CONFIG_KEYS); this returns the
    real values, since e.g. `set_config_value` needs them unmasked to round-trip.
    """
    return _read_raw_config()


def _validate_model_entry(key: str, entry: str) -> None:
    """Best-effort typo catcher for "provider:model"-shaped config values -- NOT
    a strict format requirement: a bare name with no ':' is legitimately valid
    (mazu.llm.client._split_model treats it as an Anthropic model, matching
    MAZU_MODEL=claude-opus-4-8-style config that predates this multi-provider
    naming). Only rejects a colon that's clearly a mistake -- an empty provider
    or model half (e.g. "gpt4:" or ":claude-sonnet-5") -- or an empty entry.
    Real bug found live: council_models/council_lead accepted anything through
    the generic Config "Set a value" form with zero validation, so a typo
    silently got saved and only surfaced later as a confusing, disconnected
    failure the next time Council actually ran.
    """
    entry = entry.strip()
    if not entry:
        raise ValueError(f"'{key}' contains an empty model entry.")
    if ":" in entry:
        provider_part, _, model_part = entry.partition(":")
        if not provider_part.strip() or not model_part.strip():
            raise ValueError(
                f"'{key}' entry '{entry}' looks malformed -- expected \"provider:model\" "
                f'(e.g. "anthropic:claude-sonnet-5"), or a bare model name to default to Anthropic.'
            )


# Keys expected to hold "provider:model" (optionally comma-separated) --
# validated lightly by _validate_model_entry above before being written.
_SINGLE_MODEL_CONFIG_KEYS = {"council_lead"}
_MODEL_LIST_CONFIG_KEYS = {"council_models"}


def set_config_value(key: str, value: str) -> None:
    if key not in KNOWN_CONFIG_KEYS:
        raise ValueError(
            f"Unknown config key '{key}'. Known keys: {', '.join(sorted(KNOWN_CONFIG_KEYS))}"
        )
    if key in _SINGLE_MODEL_CONFIG_KEYS:
        _validate_model_entry(key, value)
    elif key in _MODEL_LIST_CONFIG_KEYS:
        for entry in value.split(","):
            _validate_model_entry(key, entry)
    # Real bug found via live testing: mazu-web runs Flask with threaded=True, so
    # two concurrent /api/config POSTs (or a `mazu` CLI invocation racing a
    # mazu-web request) could each read the same on-disk config, add their own
    # key to their own in-memory copy, then write back -- the second write wins
    # and silently drops the first key. The lock below serializes the whole
    # read-modify-write so no update is ever lost, matching the pattern already
    # used for checkpoints/index.json (see mazu/checkpoint/lock.py).
    with _config_lock().acquire():
        config = _read_raw_config()
        config[key] = value
        _write_config(config)


def unset_config_value(key: str) -> bool:
    """Returns False if the key wasn't set (a no-op, not an error)."""
    with _config_lock().acquire():
        config = _read_raw_config()
        if key not in config:
            return False
        del config[key]
        _write_config(config)
        return True


def _toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_config(config: dict) -> None:
    # Every value Mazu's config schema stores is a plain string (an API key or a
    # "provider:model" name) -- a minimal string-only TOML writer is enough, and
    # avoids adding a TOML-writing dependency (tomllib, stdlib since 3.11, is
    # read-only) for something this simple.
    #
    # Written atomically (temp file in the same directory + os.replace) rather than
    # path.write_text(), which opens the real file in truncate mode first and writes
    # second -- a concurrent reader (or a crash) landing in that window used to see a
    # half-written or fully-empty config.toml. os.replace() is an atomic rename on
    # both POSIX and Windows, so any reader always sees either the old file or the
    # complete new one, never something in between. Confirmed live: a burst of
    # concurrent `mazu-web` config writes previously left config.toml at 0 bytes,
    # silently losing every previously-set key (API keys, default_model, curator_*).
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {_toml_quote(str(value))}" for key, value in config.items()]
    content = "\n".join(lines) + ("\n" if lines else "")
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".toml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def local_base_url() -> str:
    """MAZU_LOCAL_BASE_URL env var, then the local_base_url config.toml key, then
    LM Studio's default -- Ollama users set either the env var or `mazu config set
    local_base_url http://localhost:11434/v1`. Its own function, not folded into
    get_default_model(), since it answers a different question (a URL, not a model).
    """
    env_value = os.environ.get("MAZU_LOCAL_BASE_URL")
    if env_value:
        return env_value
    configured = _read_raw_config().get("local_base_url")
    if configured:
        return configured
    return _LOCAL_BASE_URL_DEFAULT


def ensure_api_key(model: str | None = None) -> None:
    """Checks the API key for whichever provider `model` (or the auto-detected
    default) actually resolves to — not hardcoded to Anthropic. A DeepSeek-only or
    OpenAI-only setup works with no Anthropic key at all, as long as a model/
    MAZU_MODEL naming that provider is used (or nothing else is configured and it
    gets auto-detected from whichever key is present).
    """
    load_config()
    from mazu.llm.client import _PROVIDERS, _split_model, default_model

    provider_name, _ = _split_model(model or default_model())
    provider = _PROVIDERS.get(provider_name)

    if provider is not None and not provider.requires_api_key:
        return

    env_var = provider.api_key_env if provider is not None else "ANTHROPIC_API_KEY"

    if not os.environ.get(env_var):
        raise SystemExit(
            f"No API key found for provider '{provider_name}' (needs {env_var}).\n"
            f"Set it with: {_SET_ENV_EXAMPLE.format(var=env_var)}\n"
            "or pass --model provider:model to use a different provider you already have a key for."
        )
