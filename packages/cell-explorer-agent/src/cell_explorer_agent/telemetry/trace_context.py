"""TurnTrace — async context manager that owns the per-turn Langfuse trace.

Wraps the Langfuse v3 SDK so the rest of the agent only sees this protocol.
All redaction decisions are made once at __aenter__ based on `is_public`
and applied to every subsequent add_* call.

The root observation is opened via start_as_current_observation so that
every child observation we create inside __aenter__/__aexit__ auto-attaches
as a child of the root (and thus shares the trace).

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
    """One Langfuse trace per chat turn (v3 API).

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
        # v3 internals
        self._root_cm: Any | None = None
        self._root_obs: Any | None = None

    async def __aenter__(self) -> "TurnTrace":
        if self._client is None:
            return self
        try:
            visibility = "public" if self._is_public else "private"
            # Note: no explicit env: tag — the Langfuse SDK handles environment
            # natively (LANGFUSE_TRACING_ENVIRONMENT / ENVIRONMENT) and exposes
            # it as a first-class filter in the UI rather than a tag.
            tags = [
                f"dataset:{self._dataset_slug}",
                f"model:{self._model}",
                f"visibility:{visibility}",
            ]
            # Root observation as current — children created via
            # start_observation while we're in the with block will
            # auto-attach as descendants.
            self._root_cm = self._client.start_as_current_observation(
                name="chat-turn",
                as_type="span",
                input=redact_user_input(self._user_input, public=self._is_public),
                metadata={
                    "view_state": redact_view_state(self._view_state, public=self._is_public),
                    "dataset_slug": self._dataset_slug,
                },
            )
            self._root_obs = self._root_cm.__enter__()
            # Set trace-level fields (user/session/tags) once the root span
            # is active — these belong on the implicit trace, not the span.
            self._client.update_current_trace(
                user_id=self._user_id,
                session_id=self._thread_id,
                tags=tags,
                input=redact_user_input(self._user_input, public=self._is_public),
            )
        except Exception:
            logger.exception("TurnTrace: failed to open trace")
            self._root_cm = None
            self._root_obs = None
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Always release the root context, then propagate the caller's exc.
        if self._root_cm is not None:
            try:
                self._root_cm.__exit__(exc_type, exc, tb)
            except Exception:
                logger.exception("TurnTrace: failed to close trace")
        # Returning None (= falsy) — don't suppress caller exceptions.
        return None

    def add_generation(
        self,
        *,
        input_messages: list[dict[str, Any]],
        output_text: str,
        usage: dict[str, int],
    ) -> None:
        if self._root_obs is None:
            return
        try:
            gen_input: Any = (
                input_messages
                if self._is_public
                else redact_user_input("", public=False)
            )
            gen = self._client.start_observation(
                name="llm-call",
                as_type="generation",
                input=gen_input,
                model=self._model,
            )
            gen.update(
                output=redact_assistant_output(output_text, public=self._is_public),
                usage_details=dict(usage) if isinstance(usage, dict) else None,
            )
            gen.end()
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
        if self._root_obs is None:
            return
        try:
            level = "ERROR" if status == "error" else "DEFAULT"
            status_message: str | None = None
            if status == "error" and isinstance(result, dict) and "error" in result:
                status_message = str(result["error"])
            sp = self._client.start_observation(
                name=f"tool:{name}",
                as_type="span",
                input=redact_tool_args(name, args, public=self._is_public),
            )
            sp.update(
                output=redact_tool_result(result, public=self._is_public),
                metadata={"duration_ms": duration_ms, "status": status},
                level=level,
                status_message=status_message,
            )
            sp.end()
        except Exception:
            logger.exception("TurnTrace: failed to add tool span")

    def set_output(self, text: str) -> None:
        if self._root_obs is None:
            return
        try:
            redacted = redact_assistant_output(text, public=self._is_public)
            # Root span's output (visible on the span row).
            self._root_obs.update(output=redacted)
            # And the trace-level output (visible on the trace card in the UI).
            self._client.update_current_trace(output=redacted)
        except Exception:
            logger.exception("TurnTrace: failed to set output")
