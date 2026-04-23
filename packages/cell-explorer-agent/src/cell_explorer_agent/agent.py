"""ChatAgent — turn loop over an LLMClient with data + ui-action tools."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.events import (
    ChatEvent,
    Done,
    TextDelta,
    TokenUsage,
    ToolProgress,
    UIAction,
)
from cell_explorer_agent.llm.base import (
    LLMClient,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
)
from cell_explorer_agent.messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResult,
    ToolResultMessage,
)
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
        history: list[Message] = list(messages)
        total_usage = TokenUsage()

        while True:
            assistant_text = ""
            pending_calls: list[ToolCall] = []
            stop_reason: str | None = None

            async for ev in self.llm.stream(
                system=system,
                messages=history,
                tools=tools,
                model=self.config.llm_model,
                max_tokens=2048,
                timeout_s=self.config.llm_timeout_s,
            ):
                if isinstance(ev, LLMTextDelta):
                    assistant_text += ev.text
                    yield TextDelta(text=ev.text)
                elif isinstance(ev, LLMToolCall):
                    pending_calls.append(
                        ToolCall(id=ev.id, name=ev.name, args=ev.args)
                    )
                elif isinstance(ev, LLMStop):
                    stop_reason = ev.reason
                    total_usage = TokenUsage(
                        input_tokens=total_usage.input_tokens + ev.usage.input_tokens,
                        output_tokens=total_usage.output_tokens + ev.usage.output_tokens,
                        cache_read_tokens=total_usage.cache_read_tokens
                        + ev.usage.cache_read_tokens,
                        cache_write_tokens=total_usage.cache_write_tokens
                        + ev.usage.cache_write_tokens,
                    )

            history.append(
                AssistantMessage(text=assistant_text, tool_calls=pending_calls)
            )

            if stop_reason == "end_turn" or not pending_calls:
                yield Done(usage=total_usage)
                return

            # Execute pending tool calls (data kind only for this task).
            results: list[ToolResult] = []
            for call in pending_calls:
                tool = self.catalog.get(call.name)
                if tool is None:
                    results.append(
                        ToolResult(
                            tool_call_id=call.id,
                            content={"error": f"unknown tool {call.name!r}"},
                            is_error=True,
                        )
                    )
                    continue

                if tool.kind == "data":
                    yield ToolProgress(tool=tool.name, status="started")
                    result = await tool.func(**call.args)
                    yield ToolProgress(tool=tool.name, status="ok")
                    results.append(
                        ToolResult(tool_call_id=call.id, content=result)
                    )
                else:  # tool.kind == "ui_action"
                    yield ToolProgress(tool=tool.name, status="started")
                    result = await tool.func(**call.args)
                    if "error" in result:
                        # Validation / lookup failed — never emit the action.
                        yield ToolProgress(
                            tool=tool.name, status="error", summary=result["error"]
                        )
                        results.append(
                            ToolResult(
                                tool_call_id=call.id,
                                content=result,
                                is_error=True,
                            )
                        )
                    else:
                        yield UIAction(payload=result["payload"])
                        yield ToolProgress(tool=tool.name, status="ok")
                        results.append(
                            ToolResult(
                                tool_call_id=call.id,
                                content={"dispatched": True},
                            )
                        )

            history.append(ToolResultMessage(results=results))
