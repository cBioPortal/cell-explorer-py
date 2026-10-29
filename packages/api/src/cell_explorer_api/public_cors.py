"""CORS for the public dataset catalog, so other sites can list datasets.

Separate from ``CORS_ORIGINS`` on purpose. That setting lets an origin send
the user's cookies to every endpoint. This one only answers plain ``GET`` and
``HEAD`` requests for the catalog listings, and never allows credentials, so
a matching site reads exactly what an anonymous visitor would see.
"""

import re

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

PUBLIC_CATALOG_PATHS = frozenset({"/api/datasets", "/api/collections"})


class PublicCatalogCorsMiddleware:
    """Adds ``Access-Control-Allow-Origin`` to catalog listings for matching origins."""

    def __init__(self, app: ASGIApp, origin_regex: str) -> None:
        self.app = app
        self.origin_pattern = re.compile(origin_regex)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] not in ("GET", "HEAD")
            or scope["path"].rstrip("/") not in PUBLIC_CATALOG_PATHS
        ):
            await self.app(scope, receive, send)
            return

        origin = Headers(scope=scope).get("origin")
        if not origin or not self.origin_pattern.fullmatch(origin):
            await self.app(scope, receive, send)
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                # An origin also listed in CORS_ORIGINS already has its (credentialed) answer.
                if "access-control-allow-origin" not in headers:
                    headers["Access-Control-Allow-Origin"] = origin
                    headers.add_vary_header("Origin")
            await send(message)

        await self.app(scope, receive, send_with_cors)
