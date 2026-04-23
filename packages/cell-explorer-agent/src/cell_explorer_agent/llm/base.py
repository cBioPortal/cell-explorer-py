"""Provider-agnostic LLMClient Protocol.

Adapters (AnthropicLLMClient, FakeLLMClient) translate between the provider's
wire format and this neutral event stream.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from cell_explorer_agent.messages import Message
from cell_explorer_agent.tools.registry import Tool


@dataclass
class LLMTextDelta:
    text: str


@dataclass
class LLMToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class LLMStop:
    reason: Literal["end_turn", "tool_use", "max_tokens", "error"]
    usage: LLMUsage = field(default_factory=LLMUsage)


LLMEvent = LLMTextDelta | LLMToolCall | LLMStop


class LLMClient(Protocol):
    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[Tool],
        model: str,
        max_tokens: int,
        timeout_s: float,
    ) -> AsyncIterator[LLMEvent]: ...
