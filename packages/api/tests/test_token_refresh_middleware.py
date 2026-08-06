"""Tests for the TokenRefreshMiddleware.

The middleware injects Set-Cookie headers on responses when require_auth
has stashed new tokens on request.state. Critically, this must work for
both JSON and StreamingResponse endpoints — the latter is why the
previous BaseHTTPMiddleware implementation was replaced (it dropped late
header mutations on streaming responses, causing the chat /turns endpoint
to silently fail to rotate expired access cookies).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from cell_explorer_api.auth.dependencies import require_auth
from cell_explorer_api.auth.oidc import OidcClient
from cell_explorer_api.auth.middleware import TokenRefreshMiddleware
from cell_explorer_api.auth.models import User
from cell_explorer_api.config import Settings


def _generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _make_settings() -> Settings:
    return Settings(
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
    )


def _make_token(private_key, **overrides) -> str:
    claims = {
        "sub": "user-123",
        "name": "Test User",
        "email": "test@example.com",
        "realm_access": {"roles": ["viewer"]},
        "iss": "https://auth.example.com/realms/test-realm",
        "aud": "test-client",
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
    } | overrides
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-kid"})


@pytest.fixture()
def rsa_keys():
    return _generate_rsa_keypair()


@pytest.fixture()
def keycloak(rsa_keys):
    _, public_key = rsa_keys
    settings = _make_settings()
    client = OidcClient(settings)
    client._apply_discovery({
        "issuer": "https://auth.example.com/realms/test-realm",
        "authorization_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/auth",
        "token_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/token",
        "jwks_uri": "https://auth.example.com/realms/test-realm/protocol/openid-connect/certs",
        "end_session_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/logout",
    })
    client._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    return client


def _app_with_middleware(keycloak: OidcClient) -> FastAPI:
    app = FastAPI()
    app.state.settings = _make_settings()
    app.state.oidc = keycloak
    app.add_middleware(TokenRefreshMiddleware)

    @app.get("/json")
    async def json_endpoint(user: User = Depends(require_auth)):
        return {"sub": user.sub}

    @app.post("/stream")
    async def stream_endpoint(user: User = Depends(require_auth)):
        async def gen():
            yield b'{"hello":'
            yield b' "world"}\n'

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app


def _set_cookie_names(response) -> set[str]:
    """Extract cookie names from all Set-Cookie response headers."""
    names: set[str] = set()
    for name, value in response.headers.items():
        if name.lower() == "set-cookie":
            names.add(value.split("=", 1)[0])
    # httpx flattens multiple Set-Cookie headers into a single comma-joined
    # value in .headers; the .raw API preserves them as separate entries.
    for name_b, value_b in response.headers.raw:
        if name_b.lower() == b"set-cookie":
            names.add(value_b.decode().split("=", 1)[0])
    return names


def test_middleware_sets_cookies_on_json_response_after_refresh(keycloak, rsa_keys):
    """JSON endpoint — baseline behavior (also worked with the old impl)."""
    private_key, _ = rsa_keys
    fresh_access = _make_token(private_key)
    expired_access = _make_token(private_key, exp=int(time.time()) - 60)

    app = _app_with_middleware(keycloak)
    with patch.object(
        keycloak,
        "refresh_token",
        new=AsyncMock(return_value={"access_token": fresh_access, "refresh_token": "new-refresh"}),
    ):
        client = TestClient(app)
        client.cookies.set("cce_access", expired_access)
        client.cookies.set("cce_refresh", "some-valid-refresh-token")
        response = client.get("/json")

    assert response.status_code == 200
    names = _set_cookie_names(response)
    assert "cce_access" in names
    assert "cce_refresh" in names


def test_middleware_sets_cookies_on_streaming_response_after_refresh(keycloak, rsa_keys):
    """StreamingResponse endpoint — the regression we are fixing.

    The old BaseHTTPMiddleware silently dropped Set-Cookie headers on
    StreamingResponse. The pure ASGI replacement must emit them.
    """
    private_key, _ = rsa_keys
    fresh_access = _make_token(private_key)
    expired_access = _make_token(private_key, exp=int(time.time()) - 60)

    app = _app_with_middleware(keycloak)
    with patch.object(
        keycloak,
        "refresh_token",
        new=AsyncMock(return_value={"access_token": fresh_access, "refresh_token": "new-refresh"}),
    ):
        client = TestClient(app)
        client.cookies.set("cce_access", expired_access)
        client.cookies.set("cce_refresh", "some-valid-refresh-token")
        response = client.post("/stream")

    assert response.status_code == 200
    # Body actually streamed through (proves middleware didn't break it).
    assert response.json() == {"hello": "world"}
    names = _set_cookie_names(response)
    assert "cce_access" in names
    assert "cce_refresh" in names


def test_middleware_does_nothing_when_no_refresh_happened(keycloak, rsa_keys):
    """When the access token is still valid, no Set-Cookie should be emitted."""
    private_key, _ = rsa_keys
    valid_access = _make_token(private_key)

    app = _app_with_middleware(keycloak)
    client = TestClient(app)
    client.cookies.set("cce_access", valid_access)
    response = client.get("/json")

    assert response.status_code == 200
    names = _set_cookie_names(response)
    # No rotation occurred — these cookies should not be re-set.
    assert "cce_access" not in names
    assert "cce_refresh" not in names


def test_middleware_passes_through_non_http_scope(keycloak):
    """Lifespan/websocket scopes shouldn't trip the middleware."""
    app = _app_with_middleware(keycloak)
    # FastAPI's lifespan goes through the middleware stack on startup; if
    # the middleware tried to read scope["state"].new_access_token on a
    # lifespan scope, it would raise. Construct + tear-down the app via
    # TestClient as the lifespan invocation.
    with TestClient(app):
        pass  # successful __enter__/__exit__ means lifespan ran clean
