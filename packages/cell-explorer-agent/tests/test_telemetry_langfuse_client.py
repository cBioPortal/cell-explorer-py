import pytest

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.telemetry import langfuse_client


@pytest.fixture(autouse=True)
def reset_singleton():
    langfuse_client._reset_for_tests()
    yield
    langfuse_client._reset_for_tests()


def test_get_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    cfg = AgentConfig()
    assert langfuse_client.get(cfg) is None


def test_get_returns_client_when_configured(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    cfg = AgentConfig()
    client = langfuse_client.get(cfg)
    assert client is not None


def test_get_caches_instance(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    cfg = AgentConfig()
    assert langfuse_client.get(cfg) is langfuse_client.get(cfg)


def test_get_returns_none_when_kill_switch_off(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACE_ENABLED", "false")
    cfg = AgentConfig()
    assert langfuse_client.get(cfg) is None
