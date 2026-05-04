"""Local HTTP server that captures OAuth tokens from the browser."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable


class CallbackTimeout(Exception):
    """Raised when the OAuth callback didn't arrive in time."""


def start_callback_server(
    *,
    path: str = "/callback",
    timeout_s: float = 300,
) -> tuple[int, Callable[[], dict]]:
    """Start a one-shot HTTP server on a random free port.

    Returns `(port, wait)`:
      - `port`: the bound port number.
      - `wait()`: blocks up to `timeout_s` for a valid POST to `path` with a
        JSON body. Returns the parsed dict. Raises CallbackTimeout on timeout.

    The server shuts down automatically once `wait()` returns (success or timeout).
    """
    result: dict = {}
    result_event = threading.Event()

    def _send_cors_headers(handler: BaseHTTPRequestHandler) -> None:
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type")

    class _Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):  # noqa: N802
            # CORS preflight — browser sends this before the cross-origin POST
            # from the API's localhost:8001 success page to our localhost:<port>.
            self.send_response(204)
            _send_cors_headers(self)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.path != path:
                self.send_response(404)
                _send_cors_headers(self)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                data = json.loads(body)
            except Exception:
                self.send_response(400)
                _send_cors_headers(self)
                self.end_headers()
                return
            result.update(data)
            self.send_response(204)
            _send_cors_headers(self)
            self.end_headers()
            result_event.set()

        def log_message(self, *_args, **_kwargs):  # quiet test output
            return

    # Bind to 0 to get a random port
    bind_addr = ("127.0.0.1", 0)
    httpd = HTTPServer(bind_addr, _Handler)
    port = httpd.server_address[1]

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    def wait() -> dict:
        try:
            got = result_event.wait(timeout=timeout_s)
            if not got:
                raise CallbackTimeout(
                    f"no callback received within {timeout_s}s"
                )
            return dict(result)
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=2)

    return port, wait
