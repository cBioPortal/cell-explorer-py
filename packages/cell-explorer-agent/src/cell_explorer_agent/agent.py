"""ChatAgent — turn loop over an LLMClient with data + ui-action tools."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

_MAX_ARG_LOG_LEN = 200


def _short_args(args: dict[str, Any]) -> str:
    """Compact, length-capped representation of tool args for log lines.

    Serialize as JSON for readability; truncate so a single call with a large
    payload (gene-id list, etc.) doesn't blow out a log line. Non-JSON-able
    values fall back to repr.
    """
    try:
        s = json.dumps(args, separators=(",", ":"), default=str)
    except Exception:
        s = repr(args)
    return s if len(s) <= _MAX_ARG_LOG_LEN else s[:_MAX_ARG_LOG_LEN] + "...(truncated)"


def _strip_chart_data_for_llm(result: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the tool result with chart.data removed.

    The frontend gets the full chart payload (including data) on the
    ToolProgress wire event so it can render. The LLM only needs to know
    that a chart was attached and what type — not the underlying numbers,
    which it already has in result['data']. Stripping avoids the LLM
    seeing duplicate data and lets it write a concise summary.
    """
    chart = result.get("chart")
    if not isinstance(chart, dict) or "data" not in chart:
        return result
    stripped_chart = {k: v for k, v in chart.items() if k != "data"}
    return {**result, "chart": stripped_chart}


from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.events import (
    ChatEvent,
    Done,
    Error,
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
from cell_explorer_agent.prompt.system import build_system_prompt, format_view_state_block
from cell_explorer_agent.tools.registry import ToolCatalog
from cell_explorer_agent.telemetry import langfuse_client
from cell_explorer_agent.telemetry.trace_context import TurnTrace

logger = logging.getLogger(__name__)


@dataclass
class ChatAgent:
    llm: LLMClient
    catalog: ToolCatalog
    dataset_ctx: DatasetContext
    config: AgentConfig

    async def run(
        self,
        *,
        messages: list[Message],
        view_state: dict[str, Any] | None = None,
        telemetry_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        system = build_system_prompt(self.dataset_ctx)
        view_state_block = (
            format_view_state_block(view_state) if view_state else None
        )
        tools = self.catalog.all()
        history: list[Message] = list(messages)
        total_usage = TokenUsage()
        tool_calls_this_turn = 0
        max_calls = self.config.max_tool_calls_per_turn

        # Telemetry: open a TurnTrace if the caller supplied context.
        # Caller-supplied context (vs. agent self-deriving) keeps the agent
        # decoupled from API-layer concerns like Keycloak user.sub or
        # Dataset.is_public.
        client = (
            langfuse_client.get(self.config)
            if telemetry_context is not None
            else None
        )
        if telemetry_context is not None and messages and hasattr(messages[-1], "content"):
            user_text = messages[-1].content
        else:
            user_text = ""
        async with TurnTrace(
            client=client,
            user_id=(telemetry_context or {}).get("user_id"),
            thread_id=(telemetry_context or {}).get("thread_id", "no-thread"),
            dataset_slug=(telemetry_context or {}).get("dataset_slug", self.dataset_ctx.slug),
            is_public=(telemetry_context or {}).get("is_public", False),
            model=self.config.llm_model,
            environment=(telemetry_context or {}).get("environment", "unknown"),
            user_input=user_text,
            view_state=view_state,
        ) as trace:
            final_assistant_text_buffer: list[str] = []
            while True:
                assistant_text = ""
                pending_calls: list[ToolCall] = []
                stop_reason: str | None = None
                llm_input_snapshot = list(history)

                async for ev in self.llm.stream(
                    system=system,
                    messages=history,
                    tools=tools,
                    model=self.config.llm_model,
                    max_tokens=2048,
                    timeout_s=self.config.llm_timeout_s,
                    view_state_block=view_state_block,
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
                        # Telemetry: record this LLM call.
                        trace.add_generation(
                            input_messages=[
                                {"role": getattr(m, "__class__", type(m)).__name__, "_repr": repr(m)[:200]}
                                for m in llm_input_snapshot
                            ],
                            output_text=assistant_text,
                            usage={
                                "input_tokens": ev.usage.input_tokens,
                                "output_tokens": ev.usage.output_tokens,
                                "cache_read_tokens": ev.usage.cache_read_tokens,
                                "cache_write_tokens": ev.usage.cache_write_tokens,
                            },
                        )

                final_assistant_text_buffer.append(assistant_text)
                history.append(
                    AssistantMessage(text=assistant_text, tool_calls=pending_calls)
                )

                if stop_reason == "end_turn" or not pending_calls:
                    trace.set_output("".join(final_assistant_text_buffer))
                    yield Done(usage=total_usage, trace_id=trace.trace_id)
                    return

                if tool_calls_this_turn + len(pending_calls) > max_calls:
                    yield Error(
                        message=(
                            f"Tool-call limit ({max_calls}) reached this turn — stopping."
                        ),
                        retryable=False,
                    )
                    trace.set_output("".join(final_assistant_text_buffer))
                    yield Done(usage=total_usage, trace_id=trace.trace_id)
                    return

                results: list[ToolResult] = []
                for call in pending_calls:
                    tool_calls_this_turn += 1
                    tool = self.catalog.get(call.name)
                    if tool is None:
                        results.append(
                            ToolResult(
                                tool_call_id=call.id,
                                content={"error": f"unknown tool {call.name!r}"},
                                is_error=True,
                            )
                        )
                        trace.add_tool_span(
                            name=call.name,
                            args=call.args,
                            result={"error": f"unknown tool {call.name!r}"},
                            duration_ms=0,
                            status="error",
                        )
                        continue

                    logger.info(
                        "chat.tool.call.start tool=%s kind=%s args=%s",
                        tool.name,
                        tool.kind,
                        _short_args(call.args),
                    )
                    start_ts = time.monotonic()
                    try:
                        yield ToolProgress(
                            tool=tool.name,
                            status="started",
                            args=call.args,
                            tool_call_id=call.id,
                        )
                        if tool.wants_context:
                            result = await tool.func(
                                **call.args,
                                _context={"view_state": view_state or {}},
                            )
                        else:
                            result = await tool.func(**call.args)
                    except Exception as exc:
                        duration_ms = int((time.monotonic() - start_ts) * 1000)
                        correlation = uuid.uuid4().hex
                        logger.exception(
                            "chat.tool.call.error tool=%s duration_ms=%d correlation=%s",
                            tool.name,
                            duration_ms,
                            correlation,
                        )
                        yield ToolProgress(
                            tool=tool.name,
                            status="error",
                            summary=str(exc),
                            duration_ms=duration_ms,
                            tool_call_id=call.id,
                        )
                        results.append(
                            ToolResult(
                                tool_call_id=call.id,
                                content={"error": str(exc), "correlation_id": correlation},
                                is_error=True,
                            )
                        )
                        trace.add_tool_span(
                            name=tool.name,
                            args=call.args,
                            result={"error": str(exc), "correlation_id": correlation},
                            duration_ms=duration_ms,
                            status="error",
                        )
                        continue
                    duration_ms = int((time.monotonic() - start_ts) * 1000)
                    logger.info(
                        "chat.tool.call.end tool=%s kind=%s duration_ms=%d",
                        tool.name,
                        tool.kind,
                        duration_ms,
                    )

                    if tool.kind == "data":
                        # Forward chart hint on the wire so the frontend renders.
                        chart_for_wire = (
                            result.get("chart") if isinstance(result, dict) else None
                        )
                        yield ToolProgress(
                            tool=tool.name,
                            status="ok",
                            duration_ms=duration_ms,
                            tool_call_id=call.id,
                            chart=chart_for_wire,
                        )
                        # Strip chart.data before sending to LLM — model only
                        # needs to know a chart was attached, not duplicate
                        # values that are also present in result["data"].
                        llm_content = (
                            _strip_chart_data_for_llm(result)
                            if isinstance(result, dict)
                            else result
                        )
                        results.append(
                            ToolResult(tool_call_id=call.id, content=llm_content)
                        )
                        trace.add_tool_span(
                            name=tool.name,
                            args=call.args,
                            result=result,
                            duration_ms=duration_ms,
                            status="ok",
                        )
                    else:  # ui_action
                        if "error" in result:
                            yield ToolProgress(
                                tool=tool.name,
                                status="error",
                                summary=result["error"],
                                duration_ms=duration_ms,
                                tool_call_id=call.id,
                            )
                            results.append(
                                ToolResult(
                                    tool_call_id=call.id,
                                    content=result,
                                    is_error=True,
                                )
                            )
                            trace.add_tool_span(
                                name=tool.name,
                                args=call.args,
                                result=result,
                                duration_ms=duration_ms,
                                status="error",
                            )
                        else:
                            yield UIAction(payload=result["payload"])
                            yield ToolProgress(
                                tool=tool.name,
                                status="ok",
                                duration_ms=duration_ms,
                                tool_call_id=call.id,
                            )
                            results.append(
                                ToolResult(
                                    tool_call_id=call.id,
                                    content={"dispatched": True},
                                )
                            )
                            trace.add_tool_span(
                                name=tool.name,
                                args=call.args,
                                result={"payload": result.get("payload")},
                                duration_ms=duration_ms,
                                status="ok",
                            )

                history.append(ToolResultMessage(results=results))
