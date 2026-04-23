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
