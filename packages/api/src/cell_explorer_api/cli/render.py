"""Render ChatEvent instances to terminal lines."""

import json
import os
import sys

from cell_explorer_agent.events import (
    ChatEvent,
    Done,
    Error,
    TextDelta,
    ToolProgress,
    UIAction,
)

_DIM = "\033[2m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _wrap(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _color_enabled() else text


def render_event_line(event: ChatEvent, *, verbose: bool = False) -> str:
    """Render one event as a string. Empty string means 'do not print'."""
    if isinstance(event, TextDelta):
        return event.text
    if isinstance(event, ToolProgress):
        if event.status == "started":
            return _wrap(f"[tool: {event.tool}]", _DIM)
        if event.status == "ok":
            return _wrap(f"[tool: {event.tool} ✓]", _DIM) if verbose else ""
        # error
        summary = f": {event.summary}" if event.summary else ""
        return _wrap(f"[tool error: {event.tool}{summary}]", _RED)
    if isinstance(event, UIAction):
        payload = json.dumps(event.payload, separators=(",", ":"))
        return _wrap(f"[action: {payload}]", _YELLOW)
    if isinstance(event, Error):
        return _wrap(f"error: {event.message}", _RED)
    if isinstance(event, Done):
        if not verbose:
            return ""
        u = event.usage
        return _wrap(
            f"-- done (in={u.input_tokens} out={u.output_tokens} "
            f"cache_r={u.cache_read_tokens} cache_w={u.cache_write_tokens})",
            _DIM,
        )
    return ""
