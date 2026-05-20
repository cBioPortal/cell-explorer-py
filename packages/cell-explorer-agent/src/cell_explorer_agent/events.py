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
    # Populated on the 'started' event for the "why" panel — JSON-serializable
    # representation of the tool's input arguments.
    args: dict[str, Any] | None = None
    # Populated on the 'ok' / 'error' events. Wall-clock time from when the
    # 'started' event was emitted until completion or failure.
    duration_ms: int | None = None
    # LLM-generated tool_use_id. Same string the model emits in [t:<id>]
    # citation markers — used by the frontend to scroll/flash a citation's
    # target row in the WhyPanel. Optional so older clients don't break.
    tool_call_id: str | None = None
    # Chart hint for inline rendering. When present (only on status="ok"),
    # the frontend renders a chart inline below the tool pill. Shape:
    # {"type": "<chart_type_enum>", "data": <renderer-specific>}. Tool authors
    # decide per-result whether to include this; tools without a meaningful
    # chart simply omit it.
    chart: dict[str, Any] | None = None


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
    # Server-assigned id of the persisted assistant message. The agent itself
    # does not persist, so this is None at agent emit time; the API layer fills
    # it in just before yielding the event to the client.
    message_id: str | None = None


class ThreadOpen(BaseModel):
    """Emitted first in every /turns stream — tells the client which thread
    this turn belongs to (created or continued) and its current title."""
    type: Literal["thread_open"] = "thread_open"
    thread_id: str
    title: str


ChatEvent = TextDelta | ToolProgress | UIAction | Error | Done | ThreadOpen


def dump_event(event: ChatEvent) -> str:
    """Serialize an event to a single-line JSON string (for SSE `data:`)."""
    return event.model_dump_json()
