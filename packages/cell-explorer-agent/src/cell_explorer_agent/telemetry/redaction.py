"""Pure redaction functions for trace payloads.

These run before any data leaves the process. Each takes a `public: bool`
flag derived from `Dataset.is_public`: when True, the input passes through
unchanged; when False, content fields are replaced with a sentinel and
structural fields (sizes, names) are preserved separately.

No Langfuse dependency — keeps the redaction surface unit-testable in
isolation. The trace context calls these once at trace creation (with
the dataset's `is_public` flag) and applies the result everywhere.
"""

from __future__ import annotations

import json
from typing import Any

REDACTED_SENTINEL = "[redacted]"

# 4 KB cap on tool result bodies even on public traces — bounds trace size
# on calls like top_expressed_genes that can return large payloads. The
# truncated marker is preserved so the trace shows that data existed.
TOOL_RESULT_MAX_BYTES = 4096


def redact_user_input(text: str, *, public: bool) -> str:
    return text if public else REDACTED_SENTINEL


def redact_view_state(snap: dict[str, Any] | None, *, public: bool) -> dict[str, Any] | None:
    if snap is None:
        return None
    return snap if public else {"_redacted": "view_state"}


def redact_tool_args(name: str, args: dict[str, Any], *, public: bool) -> dict[str, Any]:
    if public:
        return args
    return {"_redacted": "tool_args", "tool": name}


def redact_tool_result(result: Any, *, public: bool) -> Any:
    if not public:
        return {"_redacted": "tool_result"}
    # Public: serialize and truncate if oversize.
    try:
        encoded = json.dumps(result, default=str)
    except Exception:
        return {"_truncated": True, "_reason": "serialization_failed"}
    if len(encoded) <= TOOL_RESULT_MAX_BYTES:
        return result
    return {"_truncated": True, "_size_bytes": len(encoded)}


def redact_assistant_output(text: str, *, public: bool) -> str:
    return text if public else REDACTED_SENTINEL
