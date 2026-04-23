"""Neutral (provider-agnostic) message types for the chat agent."""

from typing import Any, Literal

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    tool_call_id: str
    content: dict[str, Any]
    is_error: bool = False


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    text: str = ""
    tool_calls: list[ToolCall] = []


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    results: list[ToolResult]


Message = UserMessage | AssistantMessage | ToolResultMessage
