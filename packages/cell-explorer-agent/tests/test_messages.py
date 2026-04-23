import pytest
from pydantic import ValidationError

from cell_explorer_agent.messages import (
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
)


def test_user_message_roundtrip():
    m = UserMessage(content="hello")
    assert m.role == "user"
    assert m.content == "hello"
    assert m.model_dump()["role"] == "user"


def test_assistant_message_with_text_and_tool_calls():
    m = AssistantMessage(
        text="I'll check",
        tool_calls=[ToolCall(id="c1", name="get_dataset_schema", args={})],
    )
    assert m.role == "assistant"
    assert m.tool_calls[0].name == "get_dataset_schema"


def test_tool_result_message():
    m = ToolResultMessage(
        results=[ToolResult(tool_call_id="c1", content={"ok": True})]
    )
    assert m.role == "tool_result"


def test_tool_call_rejects_missing_id():
    with pytest.raises(ValidationError):
        ToolCall(name="x", args={})  # type: ignore[call-arg]
