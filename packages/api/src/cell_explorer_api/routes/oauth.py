"""OAuth server routes for MCP sign-in, mounted under /api/oauth.

The MCP SDK serves authorize/token/register/revoke and RFC 8414 metadata; this
module adds the consent page, where the website session (``cce_access``
cookie) signs the user in, and the host-root copy of the metadata.
"""

import html
import secrets
import time
from collections import deque
from urllib.parse import quote, urlparse

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from mcp.server.auth.handlers.metadata import MetadataHandler
from mcp.server.auth.provider import AuthorizeError, construct_redirect_uri
from mcp.server.auth.routes import REGISTRATION_PATH, build_metadata, cors_middleware, create_auth_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from pydantic import AnyHttpUrl
from sqlmodel import col, delete
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.routing import Mount, Route, Router
from starlette.types import ASGIApp, Receive, Scope, Send

from cell_explorer_api.auth.oauth_provider import CellExplorerOAuthProvider
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import utcnow
from cell_explorer_api.db.oauth_models import OAuthPendingAuthorization
from cell_explorer_api.services.oauth_store import (
    API_TTL,
    find_mcp_resource_by_key,
    find_mcp_resource_by_url,
    find_token,
    issue_token,
)

OAUTH_PREFIX = "/api/oauth"
TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
REGISTRATIONS_PER_HOUR = 20

_PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
    "X-Frame-Options": "DENY",
}

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:32rem;margin:4rem auto;padding:0 1rem;color:#222;line-height:1.5}}
button{{font:inherit;padding:.5rem 1.25rem;margin-right:.5rem;cursor:pointer}}
code{{word-break:break-all}}
</style>
</head><body>
{body}
</body></html>"""


def _page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(
        _PAGE.format(title=html.escape(title), body=body), status_code=status_code, headers=_PAGE_HEADERS
    )


def _error_page(message: str) -> HTMLResponse:
    return _page(
        "Sign-in failed",
        f"<h2>Sign-in failed</h2><p>{html.escape(message)}</p>"
        "<p>Start the connection again from your MCP client.</p>",
        status_code=400,
    )


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _host(url: str) -> str | None:
    """``scheme://host[:port]`` lowercased, default port dropped; None if unparseable.

    Built from ``hostname`` (never ``netloc``) so userinfo such as ``trusted@`` is dropped.
    """
    try:
        parts = urlparse(url)
        port = parts.port
    except ValueError:
        return None
    scheme, host = parts.scheme.lower(), parts.hostname
    if not scheme or not host:
        return None
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        host = f"{host}:{port}"
    return f"{scheme}://{host}"


def _login_redirect(request_id: str) -> RedirectResponse:
    next_path = f"{OAUTH_PREFIX}/consent?request={request_id}"
    return RedirectResponse(f"/api/auth/login?next={quote(next_path, safe='')}", status_code=302)


class _RegistrationLimit:
    """At most ``limit`` POSTs per client IP per ``window`` seconds, in process."""

    def __init__(self, app: ASGIApp, limit: int = REGISTRATIONS_PER_HOUR, window: float = 3600):
        self.app, self.limit, self.window = app, limit, window
        self._hits: dict[str, deque[float]] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and not self._allow(scope):
            response = JSONResponse(
                {"error": "too_many_requests", "error_description": "too many client registrations"},
                status_code=429,
                headers={"Retry-After": str(int(self.window)), "Access-Control-Allow-Origin": "*"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _allow(self, scope: Scope) -> bool:
        now = time.monotonic()
        if len(self._hits) > 10_000:  # bound memory: drop idle IPs
            self._hits = {ip: d for ip, d in self._hits.items() if d and d[-1] > now - self.window}
        client = scope.get("client")
        hits = self._hits.setdefault(client[0] if client else "", deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


def _consent_router(provider: CellExplorerOAuthProvider, public_origin: str | None) -> APIRouter:
    router = APIRouter(prefix="/oauth", tags=["oauth"], include_in_schema=False)

    def session(request: Request) -> AsyncSession:
        return AsyncSession(request.app.state.db_engine)

    async def live_pending(request: Request, request_id: str) -> OAuthPendingAuthorization | None:
        async with session(request) as db:
            pending = await db.get(OAuthPendingAuthorization, request_id)
            if pending is None or pending.expires_at <= utcnow():
                return None
            return pending

    @router.get("/consent")
    async def consent_page(request: Request, request_id: str = Query("", alias="request")) -> Response:
        if not request_id:
            return _error_page("Missing authorization request.")
        user = await optional_auth(request)
        if user is None:
            return _login_redirect(request_id)
        pending = await live_pending(request, request_id)
        if pending is None:
            return _error_page("This sign-in request is unknown or has expired.")
        client = await provider.get_client(pending.client_id)
        async with session(request) as db:
            resource = await find_mcp_resource_by_url(db, pending.params.get("resource") or "")
        if client is None or resource is None or resource.revoked_at is not None:
            return _error_page("This sign-in request is no longer valid.")

        e = html.escape
        client_name = client.client_name or client.client_id
        redirect = str(pending.params.get("redirect_uri", ""))
        redirect_host = _host(redirect) or urlparse(redirect).scheme + ":"
        return _page(
            "Allow access?",
            f"<h2>Allow {e(client_name)} to use Cell Explorer as you?</h2>"
            f"<p><strong>{e(client_name)}</strong> is asking to use "
            f"<strong>{e(resource.name)}</strong> (<code>{e(resource.url)}</code>) "
            f"signed in as <strong>{e(user.email or user.name or user.sub)}</strong>.</p>"
            f"<p>It can then see the datasets your account can see. "
            f"You will be returned to <code>{e(redirect_host)}</code>.</p>"
            f'<form method="post" action="{OAUTH_PREFIX}/consent">'
            f'<input type="hidden" name="request" value="{e(request_id)}">'
            f'<input type="hidden" name="csrf" value="{e(pending.csrf)}">'
            '<button type="submit" name="decision" value="approve">Allow</button>'
            '<button type="submit" name="decision" value="deny">Deny</button>'
            "</form>",
        )

    @router.post("/consent")
    async def consent_decision(request: Request) -> Response:
        origin = request.headers.get("origin")
        if origin is not None and (public_origin is None or _host(origin) != public_origin):
            return _error_page("This request did not come from the consent page.")
        form = await request.form()
        request_id, csrf, decision = (str(form.get(k, "")) for k in ("request", "csrf", "decision"))
        if not request_id or decision not in ("approve", "deny"):
            return _error_page("Invalid consent request.")
        user = await optional_auth(request)
        if user is None:
            return _login_redirect(request_id)

        if decision == "approve":
            try:
                return RedirectResponse(await provider.approve(request_id, csrf, user), status_code=302)
            except AuthorizeError:
                return _error_page("This sign-in request is unknown, has expired, or is no longer valid.")

        pending = await live_pending(request, request_id)
        if pending is None or not secrets.compare_digest(pending.csrf.encode(), csrf.encode()):
            return _error_page("This sign-in request is unknown or has expired.")
        async with session(request) as db:
            await db.exec(delete(OAuthPendingAuthorization).where(col(OAuthPendingAuthorization.id) == request_id))
            await db.commit()
        # The SDK validated this redirect_uri against the client at /authorize.
        target = construct_redirect_uri(
            str(pending.params["redirect_uri"]), error="access_denied", state=pending.params.get("state")
        )
        return RedirectResponse(target, status_code=302)

    @router.post("/token-exchange")
    async def token_exchange(request: Request) -> Response:
        """RFC 8693: an MCP server trades its key and a user's access token for an API token."""
        headers = {"Cache-Control": "no-store"}

        def error(status: int, code: str) -> JSONResponse:
            return JSONResponse({"error": code}, status_code=status, headers=headers)

        scheme, _, key = request.headers.get("authorization", "").partition(" ")
        form = await request.form()
        subject_token = str(form.get("subject_token", ""))
        async with session(request) as db:
            key = key.strip()
            resource = await find_mcp_resource_by_key(db, key) if scheme.lower() == "bearer" and key else None
            if resource is None:
                return error(401, "invalid_client")
            if form.get("grant_type") != TOKEN_EXCHANGE_GRANT:
                return error(400, "unsupported_grant_type")
            found = await find_token(db, subject_token, kind="access") if subject_token else None
            if found is None or found[0].audience != resource.url:
                return error(400, "invalid_grant")
            plaintext, _ = await issue_token(db, grant=found[1], kind="api", audience="api")
            subject = found[1].user_sub
        return JSONResponse(
            {
                "access_token": plaintext,
                "issued_token_type": ACCESS_TOKEN_TYPE,
                "token_type": "Bearer",
                "expires_in": API_TTL,
                "subject": subject,
            },
            headers=headers,
        )

    return router


def mount_oauth(app: FastAPI, settings: Settings) -> None:
    """Add the OAuth server routes. Call before any catch-all route."""
    issuer = settings.oauth_issuer_url
    provider = CellExplorerOAuthProvider(
        session_factory=lambda: AsyncSession(app.state.db_engine), issuer_url=issuer
    )
    app.state.oauth_provider = provider
    registration = ClientRegistrationOptions(enabled=True)
    revocation = RevocationOptions(enabled=True)

    sdk_routes = create_auth_routes(
        provider,
        issuer_url=AnyHttpUrl(issuer),
        client_registration_options=registration,
        revocation_options=revocation,
    )
    sdk_routes = [
        Route(r.path, _RegistrationLimit(r.app), methods=r.methods) if r.path == REGISTRATION_PATH else r
        for r in sdk_routes
    ]

    # RFC 8414 path-inserted metadata at the host root.
    metadata = build_metadata(AnyHttpUrl(issuer), None, registration, revocation)
    app.router.routes.append(
        Route(
            f"/.well-known/oauth-authorization-server{OAUTH_PREFIX}",
            endpoint=cors_middleware(MetadataHandler(metadata).handle, ["GET", "OPTIONS"]),
            methods=["GET", "OPTIONS"],
        )
    )
    # The consent routes come before the mount, which would otherwise claim /api/oauth/consent.
    app.include_router(_consent_router(provider, _host(settings.public_url)), prefix="/api")
    app.router.routes.append(Mount(OAUTH_PREFIX, app=Router(routes=sdk_routes)))
