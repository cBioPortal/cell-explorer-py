"""TurnTrace — async context manager that owns the per-turn Langfuse Trace.

Wraps the Langfuse SDK so the rest of the agent only sees this protocol.
All redaction decisions are made once at __aenter__ based on the
`is_public` flag and applied consistently to every subsequent add_*
call.

Failure mode: any exception inside the context manager is logged and
swallowed — chat must never fail because of telemetry.
"""

from __future__ import annotations

import logging
from typing import Any

from cell_explorer_agent.telemetry.redaction import (
    redact_assistant_output,
    redact_tool_args,
    redact_tool_result,
    redact_user_input,
    redact_view_state,
)

logger = logging.getLogger(__name__)


class TurnTrace:
    """One Langfuse Trace per chat turn.

    Usage:
        async with TurnTrace(client=..., user_id=..., ...) as trace:
            trace.add_generation(...)
            trace.add_tool_span(...)
            trace.set_output(final_text)
    """

    def __init__(
        self,
        *,
        client: Any | None,
        user_id: str | None,
        thread_id: str,
        dataset_slug: str,
        is_public: bool,
        model: str,
        environment: str,
        user_input: str,
        view_state: dict[str, Any] | None,
    ) -> None:
        self._client = client
        self._user_id = user_id or "anonymous"
        self._thread_id = thread_id
        self._dataset_slug = dataset_slug
        self._is_public = is_public
        self._model = model
        self._environment = environment
        self._user_input = user_input
        self._view_state = view_state
        self._trace: Any | None = None

    async def __aenter__(self) -> "TurnTrace":
        if self._client is None:
            return self
        try:
            visibility = "public" if self._is_public else "private"
            tags = [
                f"env:{self._environment}",
                f"dataset:{self._dataset_slug}",
                f"model:{self._model}",
                f"visibility:{visibility}",
            ]
            self._trace = self._client.trace(
                name="chat-turn",
                user_id=self._user_id,
                session_id=self._thread_id,
                tags=tags,
                metadata={
                    "view_state": redact_view_state(self._view_state, public=self._is_public),
                    "dataset_slug": self._dataset_slug,
                },
                input=redact_user_input(self._user_input, public=self._is_public),
            )
        except Exception:
            logger.exception("TurnTrace: failed to open trace")
            self._trace = None
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # We don't suppress exceptions — return None (= False) so caller sees them.
        return None

    def add_generation(
        self,
        *,
        input_messages: list[dict[str, Any]],
        output_text: str,
        usage: dict[str, int],
    ) -> None:
        if self._trace is None:
            return
        try:
            gen_input: Any
            if self._is_public:
                gen_input = input_messages
            else:
                gen_input = redact_user_input("", public=self._is_public)
            self._trace.generation(
                name="llm-call",
                model=self._model,
                input=gen_input,
                output=redact_assistant_output(output_text, public=self._is_public),
                usage_details=dict(usage) if isinstance(usage, dict) else None,
            )
        except Exception:
            logger.exception("TurnTrace: failed to add generation")

    def add_tool_span(
        self,
        *,
        name: str,
        args: dict[str, Any],
        result: Any,
        duration_ms: int,
        status: str,
    ) -> None:
        if self._trace is None:
            return
        try:
            level = "ERROR" if status == "error" else "DEFAULT"
            status_message: str | None = None
            if status == "error" and isinstance(result, dict) and "error" in result:
                status_message = str(result["error"])
            self._trace.span(
                name=f"tool:{name}",
                input=redact_tool_args(name, args, public=self._is_public),
                output=redact_tool_result(result, public=self._is_public),
                metadata={"duration_ms": duration_ms, "status": status},
                level=level,
                status_message=status_message,
            )
        except Exception:
            logger.exception("TurnTrace: failed to add tool span")

    def set_output(self, text: str) -> None:
        if self._trace is None:
            return
        try:
            self._trace.update(
                output=redact_assistant_output(text, public=self._is_public),
            )
        except Exception:
            logger.exception("TurnTrace: failed to set output")
