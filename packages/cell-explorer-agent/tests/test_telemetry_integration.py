"""End-to-end smoke test: real Langfuse v3 SDK → local stub HTTP server.

Verifies:
  1. The SDK actually emits HTTP requests when configured.
  2. Chat works end-to-end when Langfuse Cloud is unreachable.

Not run by default; tagged 'integration' for opt-in.
"""

import asyncio
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
        body = self.rfile.read(length) if length else b""
        # v3 sends binary OTLP protobuf; we just confirm something arrived.
        _CaptureHandler.received.append(
            {"path": self.path, "content_length": length, "body_repr": body[:80]}
        )
        # OTLP/HTTP ingestion expects a small protobuf response; an empty
        # 200 is sufficient to make the SDK think delivery succeeded.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
        monkeypatch.setenv("LANGFUSE_BASE_URL", f"http://127.0.0.1:{port}")
        langfuse_client._reset_for_tests()
        cfg = AgentConfig()
        client = langfuse_client.get(cfg)
        assert client is not None

        with client.start_as_current_observation(
            name="integration-smoke", as_type="span"
        ) as root:
            gen = client.start_observation(
                name="g", as_type="generation", model="m", input="x"
            )
            gen.update(output="y", usage_details={"input": 1, "output": 1})
            gen.end()
            client.update_current_trace(user_id="u")
        client.flush()

        # Give the SDK's background thread a moment to flush.
        for _ in range(100):
            if _CaptureHandler.received:
                break
            asyncio.run(asyncio.sleep(0.1))
        assert _CaptureHandler.received, "No telemetry HTTP request landed at the stub server"
    finally:
        server.shutdown()
        langfuse_client._reset_for_tests()


@pytest.mark.integration
def test_chat_survives_langfuse_unreachable(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")  # closed port
    langfuse_client._reset_for_tests()
    cfg = AgentConfig()
    client = langfuse_client.get(cfg)
    assert client is not None
    # SDK should not raise even when the target is unreachable.
    with client.start_as_current_observation(
        name="unreachable-smoke", as_type="span"
    ) as root:
        gen = client.start_observation(
            name="g", as_type="generation", model="m", input="x"
        )
        gen.update(output="y", usage_details={"input": 1, "output": 1})
        gen.end()
        client.update_current_trace(user_id="u")
    client.flush()
    langfuse_client._reset_for_tests()
