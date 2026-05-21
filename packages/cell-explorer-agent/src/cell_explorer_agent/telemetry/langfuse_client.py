"""Module-level Langfuse client singleton.

Centralizes the 'is tracing enabled?' check. Callers ask `get(config)`;
they receive the SDK client if Langfuse is configured, or `None` if
disabled. When `None`, the trace context becomes a no-op — chat works
identically to today.

This module is the only one (alongside trace_context) that imports the
Langfuse SDK directly.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from cell_explorer_agent.config import AgentConfig

_client: Any | None = None
_configured: bool = False


def _resolve_environment() -> str | None:
    """LANGFUSE_TRACING_ENVIRONMENT takes precedence over ENVIRONMENT.

    Lets you deploy with `ENVIRONMENT=development` for the app while
    overriding the Langfuse-side label (e.g. to a Langfuse-project-specific
    convention). Returns None when neither is set — the SDK then uses its
    own default (typically "default").
    """
    return os.environ.get("LANGFUSE_TRACING_ENVIRONMENT") or os.environ.get(
        "ENVIRONMENT"
    )


def _resolve_release() -> str | None:
    """Resolve the git SHA to stamp on every trace as `release`.

    Prefers GIT_SHA env (set by deploys / docker-compose) over a subprocess
    fallback (useful in local dev). Returns None when neither is available
    so the Langfuse SDK falls back to its own default.
    """
    sha = os.environ.get("GIT_SHA")
    if sha:
        return sha.strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def get(config: AgentConfig) -> Any | None:
    """Return the cached Langfuse client, or None if tracing is disabled.

    First call after a config change constructs the client lazily. Subsequent
    calls return the cached instance. Tests should call `_reset_for_tests()`.
    """
    global _client, _configured
    if _configured:
        return _client

    _configured = True
    if not config.langfuse_trace_enabled:
        _client = None
        return None

    from langfuse import Langfuse

    kwargs: dict[str, Any] = {
        "public_key": config.langfuse_public_key,
        "secret_key": config.langfuse_secret_key,
        "base_url": config.langfuse_base_url,
    }
    env = _resolve_environment()
    if env is not None:
        kwargs["environment"] = env
    release = _resolve_release()
    if release is not None:
        kwargs["release"] = release
    _client = Langfuse(**kwargs)
    return _client


def flush() -> None:
    """Flush any queued traces. Call at shutdown."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            # Best-effort — never raise from shutdown.
            pass


def _reset_for_tests() -> None:
    """Test-only: clear the singleton so the next get() rebuilds."""
    global _client, _configured
    _client = None
    _configured = False
