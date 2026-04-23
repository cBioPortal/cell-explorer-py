from cell_explorer_agent.llm.anthropic import (
    neutral_tools_to_anthropic,
    neutral_messages_to_anthropic,
)
from cell_explorer_agent.messages import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cell_explorer_agent.tools.registry import Tool


async def _noop(**_):
    return {}


def test_tool_translation():
    t = Tool(
        name="get_dataset_schema",
        kind="data",
        description="Return shape.",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        func=_noop,
    )
    out = neutral_tools_to_anthropic([t])
    assert out[0]["name"] == "get_dataset_schema"
    assert out[0]["description"] == "Return shape."
    assert out[0]["input_schema"] == t.args_schema


def test_user_message_translation():
    out = neutral_messages_to_anthropic([UserMessage(content="hi")])
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_with_tool_calls():
    msg = AssistantMessage(
        text="checking", tool_calls=[ToolCall(id="c1", name="foo", args={"x": 1})]
    )
    out = neutral_messages_to_anthropic([msg])
    assert out[0]["role"] == "assistant"
    blocks = out[0]["content"]
    assert {"type": "text", "text": "checking"} in blocks
    tu = next(b for b in blocks if b["type"] == "tool_use")
    assert tu["id"] == "c1"
    assert tu["name"] == "foo"
    assert tu["input"] == {"x": 1}


def test_tool_result_translation():
    msg = ToolResultMessage(
        results=[ToolResult(tool_call_id="c1", content={"ok": True})]
    )
    out = neutral_messages_to_anthropic([msg])
    assert out[0]["role"] == "user"
    tr = out[0]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "c1"
    assert "ok" in tr["content"]


from unittest.mock import patch

from cell_explorer_agent.llm.anthropic import AnthropicLLMClient


def test_construct_anthropic_transport(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    with patch("anthropic.AsyncAnthropic") as MockA:
        AnthropicLLMClient(transport="anthropic")
        MockA.assert_called_once()


def test_construct_bedrock_transport():
    with patch("anthropic.AsyncAnthropicBedrock") as MockB:
        AnthropicLLMClient(transport="bedrock")
        MockB.assert_called_once()


def test_construct_vertex_transport():
    with patch("anthropic.AsyncAnthropicVertex") as MockV:
        AnthropicLLMClient(transport="vertex")
        MockV.assert_called_once()


def test_unknown_transport_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown transport"):
        AnthropicLLMClient(transport="nope")  # type: ignore[arg-type]
