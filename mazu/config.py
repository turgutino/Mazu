import os
import platform
import tomllib
from pathlib import Path

CONFIG_PATH = Path.home() / ".mazu" / "config.toml"

# `export VAR=...` is bash/zsh syntax and does nothing useful if pasted into
# Windows cmd.exe (it either errors or silently sets a literal variable named
# "export" nobody reads) -- show the syntax that actually works in the user's shell.
_SET_ENV_EXAMPLE = "set {var}=..." if platform.system() == "Windows" else "export {var}=..."


def load_config() -> dict:
    """Load ~/.mazu/config.toml if present. Env vars always take priority.
    Only Anthropic's key is supported via this file for now; other providers
    (OpenAI, DeepSeek) are configured via their own env vars.
    """
    config: dict = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                config = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            print(f"[config] warning: {CONFIG_PATH} is malformed ({e}); ignoring it.")
            config = {}
    if not os.environ.get("ANTHROPIC_API_KEY") and config.get("api_key"):
        os.environ["ANTHROPIC_API_KEY"] = config["api_key"]
    return config


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
