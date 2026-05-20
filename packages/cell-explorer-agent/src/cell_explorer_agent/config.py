"""Agent configuration loaded from environment variables."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """All tunable limits and LLM selection. Prefix: CHAT_."""

    model_config = SettingsConfigDict(env_prefix="CHAT_", case_sensitive=False, populate_by_name=True)

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

    # Langfuse observability (Phase 1). Keys absent → tracing disabled.
    # When keys present, defaults to enabled; set LANGFUSE_TRACE_ENABLED=false
    # to kill tracing without rotating keys (incident response).
    # NOTE: these vars are NOT prefixed with CHAT_ — they follow Langfuse's
    # standard env var names so the SDK can pick them up directly if needed.
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(
        default="https://us.cloud.langfuse.com", alias="LANGFUSE_BASE_URL"
    )
    # The bool field has no default; the property below resolves it.
    langfuse_trace_enabled_raw: str | None = Field(
        default=None, alias="LANGFUSE_TRACE_ENABLED"
    )

    @property
    def langfuse_trace_enabled(self) -> bool:
        """True iff Langfuse keys are configured AND the soft kill-switch isn't set."""
        if self.langfuse_public_key is None or self.langfuse_secret_key is None:
            return False
        if self.langfuse_trace_enabled_raw is None:
            return True
        return self.langfuse_trace_enabled_raw.strip().lower() not in {"false", "0", "no", "off"}
