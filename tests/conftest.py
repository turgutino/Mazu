import pytest


@pytest.fixture(autouse=True)
def _isolate_mazu_config_path(tmp_path, monkeypatch):
    """Global safety net: no test in this suite should ever read or write the real
    developer's ~/.mazu/config.toml. mazu.config.config_path() is a live function
    (not a frozen module-level constant) specifically so every test can be redirected
    here automatically, not just the ones that happen to remember to isolate it --
    a real config file on a real machine got polluted with test data once already
    before this fixture existed (config_path() was a frozen `Path.home()` constant
    at the time, evaluated once at import before any test could monkeypatch HOME).
    """
    monkeypatch.setattr("mazu.config.config_path", lambda: tmp_path / "config.toml")


@pytest.fixture(autouse=True)
def _isolate_curator_env(monkeypatch):
    """mazu.curator.client.resolve_curator_provider() deliberately writes
    MAZU_CURATOR_API_KEY into the real os.environ (a Provider's _get_client() reads
    it lazily, same as every other provider) -- necessary in production (one short
    `mazu curator run` process), but without cleanup it leaks across tests sharing
    this interpreter. monkeypatch.delenv here reverts automatically at teardown,
    same mechanism _isolate_mazu_config_path already relies on.
    """
    monkeypatch.delenv("MAZU_CURATOR_API_KEY", raising=False)
    import mazu.curator.client as _curator_client_module

    _curator_client_module._providers.clear()
