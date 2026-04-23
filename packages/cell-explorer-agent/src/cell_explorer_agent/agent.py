"""ChatAgent — turn loop over an LLMClient with data + ui-action tools.

This initial implementation handles text-only turns. Tool-call branches are
added in later tasks (data tools: Task 18; ui-action tools: Task 19; errors +
caps + cancellation: Task 20).
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.events import (
    ChatEvent,
    Done,
    TextDelta,
    TokenUsage,
)
from cell_explorer_agent.llm.base import (
    LLMClient,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
)
from cell_explorer_agent.messages import AssistantMessage, Message
from cell_explorer_agent.prompt.dataset_context import DatasetContext
from cell_explorer_agent.prompt.system import build_system_prompt
from cell_explorer_agent.tools.registry import ToolCatalog


@dataclass
class ChatAgent:
    llm: LLMClient
    catalog: ToolCatalog
    dataset_ctx: DatasetContext
    config: AgentConfig

    async def run(self, *, messages: list[Message]) -> AsyncIterator[ChatEvent]:
        system = build_system_prompt(self.dataset_ctx)
        tools = self.catalog.all()
        current = list(messages)

        assistant_text = ""

        async for ev in self.llm.stream(
            system=system,
            messages=current,
            tools=tools,
            model=self.config.llm_model,
            max_tokens=2048,
            timeout_s=self.config.llm_timeout_s,
        ):
            if isinstance(ev, LLMTextDelta):
                assistant_text += ev.text
                yield TextDelta(text=ev.text)
            elif isinstance(ev, LLMToolCall):
                # Tool handling in later tasks.
                raise NotImplementedError(
                    "tool calls handled in Task 18"
                )
            elif isinstance(ev, LLMStop):
                yield Done(
                    usage=TokenUsage(
                        input_tokens=ev.usage.input_tokens,
                        output_tokens=ev.usage.output_tokens,
                        cache_read_tokens=ev.usage.cache_read_tokens,
                        cache_write_tokens=ev.usage.cache_write_tokens,
                    )
                )
                # Append completed assistant turn for hypothetical future iterations
                current.append(AssistantMessage(text=assistant_text))
                return
