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
