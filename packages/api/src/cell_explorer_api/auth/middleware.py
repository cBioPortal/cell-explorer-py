"""Token refresh ASGI middleware.

When `require_auth` refreshes an expired access token it stashes the new
access/refresh tokens on `request.state`. This middleware intercepts the
response-start ASGI message and injects matching Set-Cookie headers so the
browser receives the rotated pair regardless of which endpoint triggered
the refresh.

Implemented as a pure ASGI middleware rather than the BaseHTTPMiddleware
convenience class because the latter buffers/wraps StreamingResponse in
ways that drop late header mutations — see
https://github.com/encode/starlette/issues/919. The visible symptom was
the chat POST /turns endpoint (a StreamingResponse) failing to refresh
expired access cookies, leading to 401 'Session expired' on the next
turn after the 5-minute access TTL elapsed.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class TokenRefreshMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def wrapped_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Starlette 1.0+ stores scope state as a plain dict; the
                # Request.state attribute wrapper reads/writes that dict via
                # __getattr__/__setattr__. Read it as a dict here.
                state = scope.get("state") or {}
                new_access = state.get("new_access_token")
                new_refresh = state.get("new_refresh_token")
                if new_access and new_refresh:
                    # Lazy-import to avoid a circular reference between
                    # auth and routes packages.
                    from cell_explorer_api.routes.auth import _set_token_cookies

                    request = Request(scope, receive)
                    tmp = Response()
                    _set_token_cookies(request, tmp, new_access, new_refresh)
                    # ASGI header list is technically an iterable; coerce
                    # to a mutable list before extending.
                    headers = list(message["headers"])
                    for name, value in tmp.raw_headers:
                        if name.lower() == b"set-cookie":
                            headers.append((name, value))
                    message["headers"] = headers
            await send(message)

        await self.app(scope, receive, wrapped_send)
