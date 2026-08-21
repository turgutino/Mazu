import pytest

from mazu.config import list_config
from mazu.curator.store import CuratorStore, new_curator_run_id
from mazu.curator.tools.config_tools import CURATOR_SETTABLE_CONFIG_KEYS, make_curator_config_tools


@pytest.fixture
def curator_store(tmp_path):
    s = CuratorStore(tmp_path / "curator.db")
    yield s
    s.close()


def _tools(curator_store, dry_run=False):
    run_id = new_curator_run_id()
    tools = {t.name: t for t in make_curator_config_tools(curator_store, run_id, dry_run)}
    return tools, run_id


def test_set_config_allows_allowlisted_key(curator_store):
    tools, run_id = _tools(curator_store)
    result = tools["set_config"].handler({"key": "default_model", "value": "deepseek:deepseek-chat", "rationale": "cheaper and just as effective here"})
    assert not result.is_error
    assert list_config()["default_model"] == "deepseek:deepseek-chat"


def test_set_config_rejects_non_allowlisted_key(curator_store):
    tools, run_id = _tools(curator_store)
    result = tools["set_config"].handler({"key": "anthropic_api_key", "value": "sk-hijacked", "rationale": "x"})
    assert result.is_error
    assert "anthropic_api_key" not in list_config()


def test_set_config_rejects_curator_model_itself():
    """The self-escalation guard: Curator must never be able to change its own
    model/key/enablement through the generic config tool."""
    assert "curator_model" not in CURATOR_SETTABLE_CONFIG_KEYS
    assert "curator_api_key" not in CURATOR_SETTABLE_CONFIG_KEYS
    assert "curator_enabled" not in CURATOR_SETTABLE_CONFIG_KEYS


def test_set_config_dry_run_does_not_change_anything(curator_store):
    tools, run_id = _tools(curator_store, dry_run=True)
    result = tools["set_config"].handler({"key": "default_model", "value": "deepseek:deepseek-chat", "rationale": "x"})
    assert "dry-run" in result.content
    assert "default_model" not in list_config()


def test_set_config_logs_a_reversal_hint(curator_store):
    tools, run_id = _tools(curator_store)
    tools["set_config"].handler({"key": "router_suggestions", "value": "false", "rationale": "noisy"})
    entries = curator_store.log_for_run(run_id)
    assert len(entries) == 1
    assert entries[0]["reversal_hint"] == "mazu config unset router_suggestions"


def test_get_config_omits_secrets(curator_store):
    from mazu.config import set_config_value

    set_config_value("anthropic_api_key", "sk-secret")
    set_config_value("default_model", "anthropic:claude-sonnet-5")
    tools, _ = _tools(curator_store)
    result = tools["get_config"].handler({})
    assert "sk-secret" not in result.content
    assert "default_model" in result.content
