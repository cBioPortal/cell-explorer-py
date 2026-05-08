"""Agent configuration loaded from environment variables."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """All tunable limits and LLM selection. Prefix: CHAT_."""

    model_config = SettingsConfigDict(env_prefix="CHAT_", case_sensitive=False)

    llm_transport: Literal["anthropic", "bedrock", "vertex"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"

    tool_result_max_bytes: int = Field(default=32_768, ge=1)
    filter_ids_max: int = Field(default=100_000, ge=1)
    gene_scan_concurrency: int = Field(default=32, ge=1, le=200)
    max_tool_calls_per_turn: int = Field(default=8, ge=1)

    daily_msg_budget: int = Field(default=100, ge=1)
    daily_token_budget: int = Field(default=500_000, ge=1)

    # View-control ui_action tools (Plan 2 view-config redesign): set_viewport,
    # set_summary_context, set_gene_label_column, set_render_controls. Set
    # CHAT_EXPERIMENTAL_VIEW_TOOLS=false to disable them.
    experimental_view_tools: bool = Field(default=True)

    llm_timeout_s: float = Field(default=60, gt=0)
    turn_timeout_s: float = Field(default=180, gt=0)
