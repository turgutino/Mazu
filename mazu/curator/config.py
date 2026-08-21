"""Curator's own configuration -- deliberately independent of mazu.config's main
model resolution (default_model()/ensure_api_key()/load_config()). Curator reads
its own env vars and its own config.toml keys directly; it never calls
load_config() (whose entire purpose is injecting the *shared* provider keys) and
nothing here is ever consulted by the main model's resolution chain.
"""
import os

from mazu.config import list_config

CURATOR_API_KEY_ENV = "MAZU_CURATOR_API_KEY"


def curator_api_key() -> str | None:
    """MAZU_CURATOR_API_KEY env var wins (matches every other Mazu env-var-wins-
    over-config convention), then the curator_api_key config.toml key."""
    return os.environ.get(CURATOR_API_KEY_ENV) or list_config().get("curator_api_key")


def curator_model() -> str | None:
    """'provider:model', e.g. 'anthropic:claude-haiku-4-5'. No default -- Curator
    must be explicitly pointed at a model, same as it must be explicitly given a key;
    an implicit default here could silently start spending money on a model the user
    never chose."""
    return os.environ.get("MAZU_CURATOR_MODEL") or list_config().get("curator_model")


def curator_base_url() -> str | None:
    """Only consulted for curator_model='local:...' -- a custom OpenAI-compatible
    endpoint, mirroring mazu.config.local_base_url()'s purpose for the main model."""
    return os.environ.get("MAZU_CURATOR_BASE_URL") or list_config().get("curator_base_url")


def curator_enabled() -> bool:
    """Default True once configured -- `mazu curator disable` flips this off without
    discarding the stored key/model, so re-enabling doesn't require re-entering
    them. Distinct from curator_configured(): a fully configured Curator can still
    be disabled, and `mazu curator status` reports the two independently."""
    return list_config().get("curator_enabled", "true").lower() != "false"


def curator_configured() -> bool:
    """True only once Curator has both a key and a model -- the two things it needs
    to make its own, independent LLM calls. `mazu curator run` on an unconfigured
    Curator must be a complete no-op (zero API calls, zero files written) -- every
    caller should check this (or catch CuratorNotConfigured from curator.client)
    before doing anything else."""
    return curator_api_key() is not None and curator_model() is not None
