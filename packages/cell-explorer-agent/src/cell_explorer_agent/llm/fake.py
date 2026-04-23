"""FakeLLMClient for tests."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from cell_explorer_agent.llm.base import LLMClient, LLMEvent
from cell_explorer_agent.messages import Message
from cell_explorer_agent.tools.registry import Tool


@dataclass
class Script:
    events: list[LLMEvent]
    # Recorded after a call completes (for assertions in tests):
    called_with_system: str = ""
    called_with_message_count: int = 0
    called_with_tool_names: list[str] = field(default_factory=list)


@dataclass
class FakeLLMClient(LLMClient):
    scripts: list[Script]

    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[Tool],
        model: str,
        max_tokens: int,
        timeout_s: float,
    ) -> AsyncIterator[LLMEvent]:
        assert self.scripts, "FakeLLMClient: no more scripts"
        s = self.scripts.pop(0)
        s.called_with_system = system
        s.called_with_message_count = len(messages)
        s.called_with_tool_names = [t.name for t in tools]

        async def gen() -> AsyncIterator[LLMEvent]:
            for e in s.events:
                yield e

        return gen()
