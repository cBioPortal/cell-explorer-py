"""Authentication endpoints."""

import re
import secrets
import time

import jwt as _jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from cell_explorer_api.auth.dependencies import require_auth
from cell_explorer_api.auth.oidc import OidcClient
from cell_explorer_api.auth.models import User

router = APIRouter(prefix="/auth", tags=["auth"])

_LOCALHOST_REDIRECT_RE = re.compile(r"^http://(localhost|127\.0\.0\.1):\d{1,5}(/.*)?$")
_CLI_STATE_EXP_SECONDS = 600  # 10 minutes

_CLI_SUCCESS_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Logged in</title>
<style>body{{font-family:system-ui,sans-serif;text-align:center;padding:4em;color:#333}}</style>
</head><body>
<h2>Successfully logged in</h2>
<p>You can close this tab and return to your terminal.</p>
<script>
  fetch({redirect_uri!r}, {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({tokens_json})
  }}).then(() => {{
    document.body.innerHTML += "<p style='color:#080'>Terminal ready.</p>";
  }}).catch(() => {{
    document.body.innerHTML += "<p style='color:#c00'>Could not reach the CLI. Is it still running?</p>";
  }});
</script>
</body></html>"""


def _callback_uri(request: Request) -> str:
    """Build the OAuth callback URI, respecting proxy headers."""
    url = request.url_for("callback")
    forwarded_host = request.headers.get("x-forwarded-host")
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_host or forwarded_proto:
        host_parts = forwarded_host.split(":") if forwarded_host else []
        hostname = host_parts[0] if host_parts else url.hostname
        port = int(host_parts[1]) if len(host_parts) > 1 else None
        url = url.replace(
            hostname=hostname,
            port=port,
            scheme=forwarded_proto if forwarded_proto else url.scheme,
        )
    return str(url)


def _cookie_defaults(request: Request) -> dict:
    """Cookie settings — secure when the request arrived over HTTPS."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return {
        "httponly": True,
        "secure": proto == "https",
        "samesite": "lax",
        "path": "/api",
    }


def _require_auth_enabled(request: Request) -> None:
    """Raise 501 if auth is not configured."""
    if not request.app.state.settings.auth_enabled:
        raise HTTPException(status_code=501, detail="Authentication is not configured")


def _require_cli_state_secret(request: Request) -> str:
    secret = request.app.state.settings.cli_state_secret
    if not secret:
        raise HTTPException(
            status_code=501,
            detail="CLI_STATE_SECRET is not configured; CLI login is disabled",
        )
    return secret


def _sign_cli_state(secret: str, redirect_uri: str) -> str:
    now = int(time.time())
    return _jwt.encode(
        {
            "cli_flow": True,
            "redirect_uri": redirect_uri,
            "nonce": secrets.token_urlsafe(16),
            "iat": now,
            "exp": now + _CLI_STATE_EXP_SECONDS,
        },
        secret,
        algorithm="HS256",
    )


def _decode_cli_state(state: str, secret: str) -> dict | None:
    """Return decoded state dict if this is a valid CLI-flow state; else None.

    Raises HTTPException(400) if the state decodes but is expired.
    """
    try:
        decoded = _jwt.decode(state, secret, algorithms=["HS256"])
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="CLI login state expired")
    except _jwt.InvalidTokenError:
        return None
    if decoded.get("cli_flow") is not True:
        return None
    return decoded


def _set_token_cookies(request: Request, response: Response, access_token: str, refresh_token: str) -> None:
    """Set access and refresh token cookies on a response."""
    defaults = _cookie_defaults(request)
    settings = request.app.state.settings
    response.set_cookie("cce_access", access_token, max_age=settings.access_cookie_max_age, **defaults)
    response.set_cookie("cce_refresh", refresh_token, max_age=settings.refresh_cookie_max_age, **defaults)


def _clear_token_cookies(response: Response) -> None:
    """Clear access and refresh token cookies."""
    response.delete_cookie("cce_access", path="/api")
    response.delete_cookie("cce_refresh", path="/api")


@router.get("/login")
async def login(request: Request):
    """Redirect to Keycloak login page."""
    _require_auth_enabled(request)
    oidc: OidcClient = request.app.state.oidc
    state = secrets.token_urlsafe(32)
    redirect_uri = _callback_uri(request)
    auth_url = oidc.authorization_url(redirect_uri=redirect_uri, state=state)
    response = RedirectResponse(url=auth_url, status_code=307)
    defaults = _cookie_defaults(request)
    response.set_cookie("cce_state", state, max_age=600, httponly=True, secure=defaults["secure"], samesite="lax", path="/api/auth")
    return response


@router.get("/cli-login")
async def cli_login(request: Request, redirect_uri: str):
    """Browser OAuth flow for the CLI.

    The CLI passes its localhost callback URL. We validate it, sign it into a
    state JWT, and redirect to Keycloak. The existing /callback endpoint will
    detect the CLI flow on the way back.
    """
    _require_auth_enabled(request)
    secret = _require_cli_state_secret(request)

    if not _LOCALHOST_REDIRECT_RE.match(redirect_uri):
        raise HTTPException(
            status_code=400,
            detail="redirect_uri must be a localhost HTTP URL "
                   "(http://localhost:<port>/... or http://127.0.0.1:<port>/...)",
        )

    signed_state = _sign_cli_state(secret, redirect_uri)
    oidc: OidcClient = request.app.state.oidc
    api_callback_uri = _callback_uri(request)
    auth_url = oidc.authorization_url(redirect_uri=api_callback_uri, state=signed_state)
    return RedirectResponse(url=auth_url, status_code=307)


@router.get("/callback")
async def callback(request: Request, code: str, state: str):
    """Handle Keycloak callback — exchange code for tokens."""
    _require_auth_enabled(request)
    oidc: OidcClient = request.app.state.oidc

    # Try CLI flow first (state is a signed JWT).
    settings = request.app.state.settings
    if settings.cli_state_secret:
        cli_state = _decode_cli_state(state, settings.cli_state_secret)
        if cli_state is not None:
            redirect_uri = cli_state["redirect_uri"]
            tokens = await oidc.exchange_code(code, _callback_uri(request))
            import json as _json
            tokens_payload = {
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "expires_in": tokens.get("expires_in", 300),
            }
            html = _CLI_SUCCESS_HTML.format(
                redirect_uri=redirect_uri,
                tokens_json=_json.dumps(tokens_payload),
            )
            return Response(content=html, media_type="text/html", status_code=200)

    # Browser flow (existing path, unchanged semantics).
    expected_state = request.cookies.get("cce_state")
    if not expected_state or state != expected_state:
        return Response(status_code=400, content="Invalid state parameter")
    redirect_uri = _callback_uri(request)
    tokens = await oidc.exchange_code(code, redirect_uri)
    response = RedirectResponse(url="/", status_code=302)
    _set_token_cookies(request, response, tokens["access_token"], tokens["refresh_token"])
    response.delete_cookie("cce_state", path="/api/auth")
    return response


@router.get("/me", response_model=User)
async def me(user: User = Depends(require_auth)):
    """Return the current user's identity."""
    return user.model_dump()


@router.post("/logout")
async def logout(request: Request):
    """Clear auth cookies."""
    _require_auth_enabled(request)
    response = Response(status_code=200)
    _clear_token_cookies(response)
    return response


@router.post("/refresh")
async def refresh(request: Request):
    """Rotate session cookies using the refresh cookie.

    Idempotent endpoint used by the frontend's proactive-refresh timer to
    keep the access cookie fresh before its TTL expires. Avoids the
    rotation race that happens when multiple parallel requests all try to
    refresh after expiry. Returns 401 if no refresh cookie or if Keycloak
    rejects the refresh.
    """
    _require_auth_enabled(request)
    refresh_token = request.cookies.get("cce_refresh")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh cookie")
    oidc: OidcClient = request.app.state.oidc
    try:
        tokens = await oidc.refresh_token(refresh_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Refresh failed")
    response = Response(status_code=204)
    _set_token_cookies(
        request,
        response,
        tokens["access_token"],
        tokens.get("refresh_token", refresh_token),
    )
    return response


@router.post("/token-exchange", response_model=User)
async def token_exchange(request: Request):
    """Exchange an external access token for session cookies."""
    _require_auth_enabled(request)
    oidc: OidcClient = request.app.state.oidc
    body = await request.json()
    access_token = body.get("accessToken")
    if not access_token:
        return Response(status_code=400, content="accessToken required")
    try:
        user = oidc.decode_token(access_token)
    except Exception:
        return Response(status_code=401, content="Invalid token")
    response = Response(status_code=200)
    defaults = _cookie_defaults(request)
    settings = request.app.state.settings
    response.set_cookie("cce_access", access_token, max_age=settings.access_cookie_max_age, **defaults)
    response.body = user.model_dump_json().encode()
    response.media_type = "application/json"
    return response
