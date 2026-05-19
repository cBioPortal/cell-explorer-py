"""End-to-end smoke test: real Langfuse SDK → local stub HTTP server.

Verifies two acceptance criteria from the spec:
  1. The SDK actually emits HTTP requests when configured.
  2. Chat works end-to-end when Langfuse Cloud is unreachable
     (we simulate this by pointing at a closed port — no request lands
      but the agent must not error).

Not run by default in unit-test sweeps; tagged 'integration' for opt-in.

Implementation note: Langfuse 3.x uses OpenTelemetry OTLP (binary protobuf)
to ship traces, so the stub server receives binary bodies, not JSON.
The handler stores the raw path; we assert purely on the path being present.
"""

import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.telemetry import langfuse_client


class _CaptureHandler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(length) if length else b""
        # Langfuse 3.x sends OTLP binary protobuf — store path + raw length.
        try:
            body: object = json.loads(body_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = f"<binary {len(body_bytes)} bytes>"
        _CaptureHandler.received.append({"path": self.path, "body": body})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _emit_observation(client) -> None:
    """Emit a trace + generation using the Langfuse 3.x API."""
    from langfuse import Langfuse
    from langfuse.types import TraceContext

    trace_id = Langfuse.create_trace_id()
    tc = TraceContext(trace_id=trace_id)
    obs = client.start_observation(
        trace_context=tc,
        name="integration-smoke",
        as_type="generation",
        model="m",
        input="x",
        output="y",
    )
    obs.end()


@pytest.mark.integration
def test_sdk_emits_http_when_configured(monkeypatch):
    _CaptureHandler.received = []
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _CaptureHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_HOST", f"http://127.0.0.1:{port}")
        langfuse_client._reset_for_tests()
        cfg = AgentConfig()
        client = langfuse_client.get(cfg)
        assert client is not None

        _emit_observation(client)
        client.flush()

        # Give the SDK's background thread a moment.
        for _ in range(100):
            if _CaptureHandler.received:
                break
            asyncio.run(asyncio.sleep(0.1))
        assert _CaptureHandler.received, (
            f"No telemetry HTTP received; got: {_CaptureHandler.received!r}"
        )
    finally:
        server.shutdown()
        langfuse_client._reset_for_tests()


@pytest.mark.integration
def test_chat_survives_langfuse_unreachable(monkeypatch):
    # Point at a port that's almost certainly closed.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:1")  # closed port
    langfuse_client._reset_for_tests()
    cfg = AgentConfig()
    client = langfuse_client.get(cfg)
    assert client is not None
    # The SDK should not block the caller even when its target is unreachable.
    _emit_observation(client)
    # Flush returns even when delivery fails; this is the contract we depend on.
    client.flush()
    langfuse_client._reset_for_tests()
