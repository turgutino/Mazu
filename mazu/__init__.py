from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: reads whatever version is actually installed, so this
    # can never drift from the real package the way a hand-maintained string here
    # previously could (and had -- this used to be a stale hardcoded "0.1.0" that
    # nothing ever updated alongside pyproject.toml's version bumps).
    __version__ = version("mazu")
except PackageNotFoundError:
    __version__ = "dev"
