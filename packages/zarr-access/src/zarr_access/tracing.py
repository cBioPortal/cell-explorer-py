"""HTTP request tracing for zarr access.

Enabled by setting ``ZARR_ACCESS_TRACE=1`` in the environment. When on, every
aiohttp request the underlying fsspec store makes is logged with: URL path,
status code, response size (when known), and wall-time in ms.

Output destination:

- Default (no file env): per-request lines to stderr, summary at exit.
- ``ZARR_ACCESS_TRACE_FILE=/path/to.jsonl``: per-request JSONL appended to that
  file (one JSON object per line). Stderr is suppressed for per-request lines
  to keep the screen quiet during heavy queries; the exit summary still prints
  to stderr so you know how much was logged.

JSONL schema per request::

    {"ts": <float epoch>, "method": "GET", "url": "...", "path": "...",
     "status": 200, "bytes": 1234, "elapsed_ms": 340.5}

Designed for ad-hoc dev investigation. No persistent state across imports
beyond the file handle while the process runs. ``atexit`` handles cleanup.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import IO

import aiohttp


def _enabled() -> bool:
    return os.environ.get("ZARR_ACCESS_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _file_path() -> str | None:
    path = os.environ.get("ZARR_ACCESS_TRACE_FILE", "").strip()
    return path or None


@dataclass
class _Stats:
    requests: int = 0
    bytes_received: int = 0
    elapsed_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_STATS = _Stats()
_SUMMARY_REGISTERED = False
_FILE_HANDLE: IO[str] | None = None
_FILE_LOCK = threading.Lock()


def _now_s() -> float:
    try:
        return asyncio.get_event_loop().time()
    except RuntimeError:
        # Called outside an event loop (e.g. atexit summary). Fall back to
        # wall clock; the absolute value doesn't matter for elapsed math.
        return time.monotonic()


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _open_file(path: str) -> IO[str] | None:
    """Open the trace file in append mode. Returns None on failure."""
    try:
        return open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
    except OSError as exc:
        _stderr(f"[zarr] could not open trace file {path!r}: {exc}")
        return None


def _ensure_file_open() -> IO[str] | None:
    global _FILE_HANDLE
    if _FILE_HANDLE is not None:
        return _FILE_HANDLE
    path = _file_path()
    if path is None:
        return None
    with _FILE_LOCK:
        if _FILE_HANDLE is None:  # double-checked
            _FILE_HANDLE = _open_file(path)
    return _FILE_HANDLE


def _write_jsonl(record: dict) -> None:
    fh = _ensure_file_open()
    if fh is None:
        return
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with _FILE_LOCK:
        fh.write(line)


def _close_file() -> None:
    global _FILE_HANDLE
    with _FILE_LOCK:
        if _FILE_HANDLE is not None:
            try:
                _FILE_HANDLE.close()
            except OSError:
                pass
            _FILE_HANDLE = None


def _print_summary() -> None:
    if _STATS.requests == 0:
        return
    mb = _STATS.bytes_received / (1024 * 1024)
    dest = "stderr" if _file_path() is None else _file_path()
    _stderr(
        f"[zarr] summary: {_STATS.requests} requests, "
        f"{mb:.1f} MB received, {_STATS.elapsed_s:.2f}s total wall-time "
        f"(per-request log → {dest})"
    )


def _ensure_summary_registered() -> None:
    global _SUMMARY_REGISTERED
    if _SUMMARY_REGISTERED:
        return
    atexit.register(_print_summary)
    atexit.register(_close_file)
    _SUMMARY_REGISTERED = True


def _emit(record: dict, human_line: str) -> None:
    """Write a per-request entry to file (if configured) or stderr."""
    if _file_path() is not None:
        _write_jsonl(record)
    else:
        _stderr(human_line)


async def _on_request_start(session, ctx, params):
    ctx.start = _now_s()


async def _on_request_end(session, ctx, params):
    elapsed_s = _now_s() - ctx.start
    response = params.response
    size = response.content_length
    size_str = (
        f"{size / 1024:.0f}KB" if size is not None and size >= 1024
        else f"{size}B" if size is not None
        else "?"
    )
    record = {
        "ts": time.time(),
        "method": params.method,
        "url": str(params.url),
        "path": params.url.path,
        "status": response.status,
        "bytes": size,
        "elapsed_ms": round(elapsed_s * 1000, 1),
    }
    human = (
        f"[zarr] {params.method} {params.url.path} "
        f"→ {response.status} ({size_str}, {elapsed_s * 1000:.0f}ms)"
    )
    _emit(record, human)
    with _STATS.lock:
        _STATS.requests += 1
        _STATS.elapsed_s += elapsed_s
        if size is not None:
            _STATS.bytes_received += size


async def _on_request_exception(session, ctx, params):
    start = getattr(ctx, "start", None)
    elapsed_s = (_now_s() - start) if start is not None else 0.0
    exc = params.exception
    record = {
        "ts": time.time(),
        "method": params.method,
        "url": str(params.url),
        "path": params.url.path,
        "status": None,
        "bytes": None,
        "elapsed_ms": round(elapsed_s * 1000, 1),
        "error": f"{type(exc).__name__}: {exc}",
    }
    human = (
        f"[zarr] {params.method} {params.url.path} "
        f"→ EXC {type(exc).__name__}: {exc} ({elapsed_s * 1000:.0f}ms)"
    )
    _emit(record, human)
    with _STATS.lock:
        _STATS.requests += 1


def build_trace_configs() -> list[aiohttp.TraceConfig] | None:
    """Return a list of aiohttp TraceConfigs if tracing is enabled, else None.

    The result is suitable for fsspec's ``client_kwargs.trace_configs``.
    """
    if not _enabled():
        return None
    _ensure_summary_registered()
    config = aiohttp.TraceConfig()
    config.on_request_start.append(_on_request_start)
    config.on_request_end.append(_on_request_end)
    config.on_request_exception.append(_on_request_exception)
    return [config]
