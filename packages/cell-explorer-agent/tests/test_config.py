import pytest
from pydantic import ValidationError

from cell_explorer_agent.config import AgentConfig


def test_defaults():
    cfg = AgentConfig()
    assert cfg.llm_transport == "anthropic"
    assert cfg.llm_model == "claude-sonnet-4-6"
    assert cfg.tool_result_max_bytes == 32_768
    assert cfg.filter_ids_max == 100_000
    assert cfg.max_tool_calls_per_turn == 8
    assert cfg.daily_msg_budget == 100
    assert cfg.daily_token_budget == 500_000
    assert cfg.llm_timeout_s == 60
    assert cfg.turn_timeout_s == 180


def test_overrides_from_env(monkeypatch):
    monkeypatch.setenv("CHAT_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("CHAT_TOOL_RESULT_MAX_BYTES", "4096")
    cfg = AgentConfig()
    assert cfg.llm_model == "claude-haiku-4-5-20251001"
    assert cfg.tool_result_max_bytes == 4096


def test_gene_scan_concurrency_default():
    cfg = AgentConfig()
    assert cfg.gene_scan_concurrency == 32


def test_gene_scan_concurrency_from_env(monkeypatch):
    monkeypatch.setenv("CHAT_GENE_SCAN_CONCURRENCY", "64")
    cfg = AgentConfig()
    assert cfg.gene_scan_concurrency == 64


def test_gene_scan_concurrency_rejects_zero():
    with pytest.raises(ValidationError):
        AgentConfig(gene_scan_concurrency=0)


def test_gene_scan_concurrency_rejects_too_large():
    with pytest.raises(ValidationError):
        AgentConfig(gene_scan_concurrency=201)


def test_gene_scan_concurrency_accepts_max():
    cfg = AgentConfig(gene_scan_concurrency=200)
    assert cfg.gene_scan_concurrency == 200


def test_experimental_view_tools_default_is_true():
    cfg = AgentConfig()
    assert cfg.experimental_view_tools is True


def test_experimental_view_tools_disabled_via_env(monkeypatch):
    monkeypatch.setenv("CHAT_EXPERIMENTAL_VIEW_TOOLS", "false")
    cfg = AgentConfig()
    assert cfg.experimental_view_tools is False


def test_langfuse_disabled_when_keys_unset(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_TRACE_ENABLED", raising=False)
    from cell_explorer_agent.config import AgentConfig
    cfg = AgentConfig()
    assert cfg.langfuse_public_key is None
    assert cfg.langfuse_secret_key is None
    assert cfg.langfuse_trace_enabled is False


def test_langfuse_enabled_by_default_when_keys_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_TRACE_ENABLED", raising=False)
    from cell_explorer_agent.config import AgentConfig
    cfg = AgentConfig()
    assert cfg.langfuse_public_key == "pk-test"
    assert cfg.langfuse_secret_key == "sk-test"
    assert cfg.langfuse_trace_enabled is True


def test_langfuse_explicit_disable_overrides_default(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACE_ENABLED", "false")
    from cell_explorer_agent.config import AgentConfig
    cfg = AgentConfig()
    assert cfg.langfuse_trace_enabled is False


def test_langfuse_base_url_default(monkeypatch):
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    from cell_explorer_agent.config import AgentConfig
    cfg = AgentConfig()
    assert cfg.langfuse_base_url == "https://us.cloud.langfuse.com"


def test_bedrock_region_default_and_override(monkeypatch):
    assert AgentConfig().bedrock_region == "us-east-1"
    monkeypatch.setenv("CHAT_BEDROCK_REGION", "us-west-2")
    assert AgentConfig().bedrock_region == "us-west-2"
