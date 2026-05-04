"""AnthropicLLMClient: translates neutral chat types to the Anthropic SDK.

Networking is wired in Task 23. This module currently exposes two pure
translation functions unit-tested without the SDK.
"""

import json
from typing import Any

from cell_explorer_agent.messages import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from cell_explorer_agent.tools.registry import Tool


def neutral_tools_to_anthropic(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert neutral tools to Anthropic's tools=[{name, description, input_schema}] format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.args_schema,
        }
        for t in tools
    ]


def neutral_messages_to_anthropic(
    messages: list[Message],
) -> list[dict[str, Any]]:
    """Convert neutral messages to Anthropic messages= format.

    - UserMessage → {"role": "user", "content": str}
    - AssistantMessage → {"role": "assistant", "content": [text?, tool_use*]}
    - ToolResultMessage → {"role": "user", "content": [{type:tool_result, ...}*]}
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            blocks: list[dict[str, Any]] = []
            if m.text:
                blocks.append({"type": "text", "text": m.text})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.args,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        elif isinstance(m, ToolResultMessage):
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": json.dumps(r.content),
                    "is_error": r.is_error,
                }
                for r in m.results
            ]
            out.append({"role": "user", "content": blocks})
    return out


from collections.abc import AsyncIterator
from typing import Literal

from cell_explorer_agent.llm.base import (
    LLMClient,
    LLMEvent,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
)
from cell_explorer_agent.llm.retry import retry_async


Transport = Literal["anthropic", "bedrock", "vertex"]


class AnthropicLLMClient(LLMClient):
    """Anthropic SDK adapter. Supports direct/Bedrock/Vertex via transport arg."""

    def __init__(self, *, transport: Transport = "anthropic") -> None:
        import anthropic

        if transport == "anthropic":
            self._client = anthropic.AsyncAnthropic()
        elif transport == "bedrock":
            self._client = anthropic.AsyncAnthropicBedrock()
        elif transport == "vertex":
            self._client = anthropic.AsyncAnthropicVertex()
        else:
            raise ValueError(f"unknown transport: {transport!r}")
        self._transport = transport

    def stream(
        self,
        *,
        system: str,
        messages: list,
        tools: list,
        model: str,
        max_tokens: int,
        timeout_s: float,
    ) -> AsyncIterator[LLMEvent]:
        return self._stream(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    async def _stream(
        self,
        *,
        system: str,
        messages,
        tools,
        model: str,
        max_tokens: int,
        timeout_s: float,
    ) -> AsyncIterator[LLMEvent]:
        # Cached system prompt — dataset context is stable per session.
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        anthropic_tools = neutral_tools_to_anthropic(tools)
        anthropic_messages = neutral_messages_to_anthropic(messages)

        async def open_stream():
            return self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=anthropic_messages,
                tools=anthropic_tools,
                timeout=timeout_s,
            )

        # Retry on provider transport errors (connection, 5xx, rate limit).
        import anthropic

        stream_cm = await retry_async(
            open_stream,
            max_attempts=3,
            base_delay_s=1.0,
            retry_on=(
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
                anthropic.RateLimitError,
            ),
        )

        async with stream_cm as stream:
            tool_blocks: dict[str, dict] = {}
            async for chunk in stream:
                # Text deltas
                if getattr(chunk, "type", None) == "content_block_delta":
                    delta = getattr(chunk, "delta", None)
                    if getattr(delta, "type", None) == "text_delta":
                        yield LLMTextDelta(text=delta.text)
                    elif getattr(delta, "type", None) == "input_json_delta":
                        # Accumulate streamed tool-input JSON fragments.
                        idx = getattr(chunk, "index", None)
                        if idx is not None and idx in tool_blocks:
                            tool_blocks[idx]["partial_json"] = (
                                tool_blocks[idx].get("partial_json", "")
                                + delta.partial_json
                            )
                elif getattr(chunk, "type", None) == "content_block_start":
                    block = getattr(chunk, "content_block", None)
                    idx = getattr(chunk, "index", None)
                    if getattr(block, "type", None) == "tool_use" and idx is not None:
                        tool_blocks[idx] = {
                            "id": block.id,
                            "name": block.name,
                            "partial_json": "",
                        }

            # Stream ended; emit any tool calls, then stop.
            import json as _json

            for idx in sorted(tool_blocks):
                tb = tool_blocks[idx]
                try:
                    args = _json.loads(tb["partial_json"] or "{}")
                except _json.JSONDecodeError:
                    args = {}
                yield LLMToolCall(id=tb["id"], name=tb["name"], args=args)

            final = await stream.get_final_message()
            usage = LLMUsage(
                input_tokens=getattr(final.usage, "input_tokens", 0),
                output_tokens=getattr(final.usage, "output_tokens", 0),
                cache_read_tokens=getattr(final.usage, "cache_read_input_tokens", 0)
                or 0,
                cache_write_tokens=getattr(
                    final.usage, "cache_creation_input_tokens", 0
                )
                or 0,
            )
            reason_map = {
                "end_turn": "end_turn",
                "tool_use": "tool_use",
                "max_tokens": "max_tokens",
            }
            reason = reason_map.get(getattr(final, "stop_reason", "") or "", "error")
            yield LLMStop(reason=reason, usage=usage)
