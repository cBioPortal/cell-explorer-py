"""Streaming chat events produced by ChatAgent.run()."""

from typing import Any, Literal

from pydantic import BaseModel


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolProgress(BaseModel):
    type: Literal["tool_progress"] = "tool_progress"
    tool: str
    status: Literal["started", "ok", "error"]
    summary: str | None = None


class UIAction(BaseModel):
    type: Literal["ui_action"] = "ui_action"
    payload: dict[str, Any]  # partial AppConfig; validated before construction


class Error(BaseModel):
    type: Literal["error"] = "error"
    message: str
    retryable: bool = False


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Done(BaseModel):
    type: Literal["done"] = "done"
    usage: TokenUsage = TokenUsage()


ChatEvent = TextDelta | ToolProgress | UIAction | Error | Done


def dump_event(event: ChatEvent) -> str:
    """Serialize an event to a single-line JSON string (for SSE `data:`)."""
    return event.model_dump_json()
