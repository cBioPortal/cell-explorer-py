"""Authentication endpoints."""

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from cell_explorer_api.auth.dependencies import require_auth
from cell_explorer_api.auth.keycloak import KeycloakClient
from cell_explorer_api.auth.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


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
    """Cookie settings — secure only when not running on localhost."""
    hostname = request.headers.get("x-forwarded-host", request.url.hostname or "").split(":")[0]
    is_localhost = hostname in ("localhost", "127.0.0.1")
    return {
        "httponly": True,
        "secure": not is_localhost,
        "samesite": "lax",
        "path": "/api",
    }


def _require_auth_enabled(request: Request) -> None:
    """Raise 501 if auth is not configured."""
    if not request.app.state.settings.auth_enabled:
        raise HTTPException(status_code=501, detail="Authentication is not configured")


def _set_token_cookies(request: Request, response: Response, access_token: str, refresh_token: str) -> None:
    """Set access and refresh token cookies on a response."""
    defaults = _cookie_defaults(request)
    response.set_cookie("cce_access", access_token, max_age=300, **defaults)
    response.set_cookie("cce_refresh", refresh_token, max_age=28800, **defaults)


def _clear_token_cookies(response: Response) -> None:
    """Clear access and refresh token cookies."""
    response.delete_cookie("cce_access", path="/api")
    response.delete_cookie("cce_refresh", path="/api")


@router.get("/login")
async def login(request: Request):
    """Redirect to Keycloak login page."""
    _require_auth_enabled(request)
    keycloak: KeycloakClient = request.app.state.keycloak
    state = secrets.token_urlsafe(32)
    redirect_uri = _callback_uri(request)
    auth_url = keycloak.authorization_url(redirect_uri=redirect_uri, state=state)
    response = RedirectResponse(url=auth_url, status_code=307)
    defaults = _cookie_defaults(request)
    response.set_cookie("cce_state", state, max_age=600, httponly=True, secure=defaults["secure"], samesite="lax", path="/api/auth")
    return response


@router.get("/callback")
async def callback(request: Request, code: str, state: str):
    """Handle Keycloak callback — exchange code for tokens."""
    _require_auth_enabled(request)
    keycloak: KeycloakClient = request.app.state.keycloak
    expected_state = request.cookies.get("cce_state")
    if not expected_state or state != expected_state:
        return Response(status_code=400, content="Invalid state parameter")
    redirect_uri = _callback_uri(request)
    tokens = await keycloak.exchange_code(code, redirect_uri)
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


@router.post("/token-exchange", response_model=User)
async def token_exchange(request: Request):
    """Exchange an external access token for session cookies."""
    _require_auth_enabled(request)
    keycloak: KeycloakClient = request.app.state.keycloak
    body = await request.json()
    access_token = body.get("accessToken")
    if not access_token:
        return Response(status_code=400, content="accessToken required")
    try:
        user = keycloak.decode_token(access_token)
    except Exception:
        return Response(status_code=401, content="Invalid token")
    response = Response(status_code=200)
    defaults = _cookie_defaults(request)
    response.set_cookie("cce_access", access_token, max_age=300, **defaults)
    response.body = user.model_dump_json().encode()
    response.media_type = "application/json"
    return response
