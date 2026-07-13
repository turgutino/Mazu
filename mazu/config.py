import os
import platform
import tomllib
from pathlib import Path

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
KNOWN_CONFIG_KEYS = {"default_model", "api_key", *_PROVIDER_KEY_ENV_VARS}
# Keys whose stored value is a secret -- `mazu config list` masks these, never
# printing the real value.
_SECRET_CONFIG_KEYS = {"api_key", *_PROVIDER_KEY_ENV_VARS}


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


def load_config() -> dict:
    """Load ~/.mazu/config.toml if present and inject any configured API keys into
    the environment -- env vars set directly always take priority and are never
    overwritten. `api_key` is a legacy alias for `anthropic_api_key`.
    """
    config = _read_raw_config()

    legacy_anthropic_key = config.get("api_key")
    if legacy_anthropic_key and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = legacy_anthropic_key

    for config_key, env_var in _PROVIDER_KEY_ENV_VARS.items():
        value = config.get(config_key)
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value

    return config


def get_default_model() -> str | None:
    """The `default_model` set via `mazu config set default_model provider:model`, or
    None if unset. Consulted by client.default_model() after the MAZU_MODEL env var
    (which always wins, matching how config-file API keys never override env vars
    either) and before auto-detecting from whichever provider's key is present.
    """
    return _read_raw_config().get("default_model")


def list_config() -> dict:
    """Raw config exactly as stored on disk -- callers displaying this to a user
    should mask secret values themselves (see _SECRET_CONFIG_KEYS); this returns the
    real values, since e.g. `set_config_value` needs them unmasked to round-trip.
    """
    return _read_raw_config()


def set_config_value(key: str, value: str) -> None:
    if key not in KNOWN_CONFIG_KEYS:
        raise ValueError(
            f"Unknown config key '{key}'. Known keys: {', '.join(sorted(KNOWN_CONFIG_KEYS))}"
        )
    config = _read_raw_config()
    config[key] = value
    _write_config(config)


def unset_config_value(key: str) -> bool:
    """Returns False if the key wasn't set (a no-op, not an error)."""
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
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {_toml_quote(str(value))}" for key, value in config.items()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
    env_var = provider.api_key_env if provider is not None else "ANTHROPIC_API_KEY"

    if not os.environ.get(env_var):
        raise SystemExit(
            f"No API key found for provider '{provider_name}' (needs {env_var}).\n"
            f"Set it with: {_SET_ENV_EXAMPLE.format(var=env_var)}\n"
            "or pass --model provider:model to use a different provider you already have a key for."
        )
