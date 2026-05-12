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


async def _drain(stream):
    async for _ in stream:
        pass


def _mock_anthropic_stream_cm():
    """A minimal async context manager that mimics the Anthropic SDK stream
    object well enough to drive _stream() through to get_final_message()."""
    from unittest.mock import AsyncMock, MagicMock

    final = MagicMock()
    final.usage.input_tokens = 0
    final.usage.output_tokens = 0
    final.usage.cache_read_input_tokens = 0
    final.usage.cache_creation_input_tokens = 0
    final.stop_reason = "end_turn"

    stream_obj = MagicMock()
    # async iterator over chunks — yield nothing (no text, no tools)
    async def _aiter(self):
        if False:
            yield None
        return
    stream_obj.__aiter__ = _aiter
    stream_obj.get_final_message = AsyncMock(return_value=final)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=stream_obj)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


async def test_stream_emits_single_cached_block_when_no_view_state(monkeypatch):
    """Without view_state_block, system arg should be a single cached block —
    the existing cache prefix behavior must not regress."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    with patch("anthropic.AsyncAnthropic") as MockA:
        captured = {}

        def capture_stream(**kwargs):
            captured.update(kwargs)
            return _mock_anthropic_stream_cm()

        MockA.return_value.messages.stream = capture_stream
        client = AnthropicLLMClient(transport="anthropic")
        await _drain(
            client.stream(
                system="STATIC",
                messages=[UserMessage(content="hi")],
                tools=[],
                model="m",
                max_tokens=100,
                timeout_s=10,
            )
        )

    blocks = captured["system"]
    assert len(blocks) == 1
    assert blocks[0] == {
        "type": "text",
        "text": "STATIC",
        "cache_control": {"type": "ephemeral"},
    }


async def test_stream_appends_uncached_block_when_view_state_present(monkeypatch):
    """With view_state_block, system arg has 2 blocks: cached static + uncached view state.
    Caching the dynamic block would invalidate the cache every turn."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    with patch("anthropic.AsyncAnthropic") as MockA:
        captured = {}

        def capture_stream(**kwargs):
            captured.update(kwargs)
            return _mock_anthropic_stream_cm()

        MockA.return_value.messages.stream = capture_stream
        client = AnthropicLLMClient(transport="anthropic")
        await _drain(
            client.stream(
                system="STATIC",
                messages=[UserMessage(content="hi")],
                tools=[],
                model="m",
                max_tokens=100,
                timeout_s=10,
                view_state_block="VIEW",
            )
        )

    blocks = captured["system"]
    assert len(blocks) == 2
    # Static prefix stays cached
    assert blocks[0]["text"] == "STATIC"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    # View-state block has NO cache_control
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"] == "VIEW"
    assert "cache_control" not in blocks[1]
