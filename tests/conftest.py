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
