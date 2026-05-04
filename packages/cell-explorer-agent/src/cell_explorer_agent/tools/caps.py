"""Size-cap helpers for tool responses."""

import json
from typing import Any


def cap_json_bytes(payload: dict[str, Any], *, limit_bytes: int) -> dict[str, Any]:
    """Return payload if its JSON encoding fits within limit_bytes; otherwise
    return a truncation marker the LLM can use to narrow the request.
    """
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) <= limit_bytes:
        return payload
    return {
        "_truncated": True,
        "limit_bytes": limit_bytes,
        "actual_bytes": len(encoded),
        "hint": (
            "The result was larger than the per-tool-response cap. "
            "Narrow the query (e.g. smaller n, a more specific group) and retry."
        ),
    }
