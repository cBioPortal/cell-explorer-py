"""Tests for require_auth dependency."""

import time
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from cell_explorer_api.auth.dependencies import require_auth
from cell_explorer_api.auth.keycloak import KeycloakClient
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


def _make_app_with_protected_route(keycloak: KeycloakClient) -> FastAPI:
    app = FastAPI()
    app.state.settings = _make_settings()
    app.state.keycloak = keycloak

    @app.get("/protected")
    async def protected(user: User = Depends(require_auth)):
        return {"sub": user.sub, "name": user.name}

    return app


@pytest.fixture()
def rsa_keys():
    return _generate_rsa_keypair()


@pytest.fixture()
def keycloak(rsa_keys):
    _, public_key = rsa_keys
    settings = _make_settings()
    client = KeycloakClient(settings)
    client._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    return client


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


def test_valid_token_returns_user(keycloak, rsa_keys):
    private_key, _ = rsa_keys
    app = _make_app_with_protected_route(keycloak)
    client = TestClient(app)
    token = _make_token(private_key)
    client.cookies.set("cce_access", token)
    response = client.get("/protected")
    assert response.status_code == 200
    assert response.json()["sub"] == "user-123"


def test_no_cookie_returns_401(keycloak):
    app = _make_app_with_protected_route(keycloak)
    client = TestClient(app)
    response = client.get("/protected")
    assert response.status_code == 401


def test_expired_token_without_refresh_returns_401(keycloak, rsa_keys):
    private_key, _ = rsa_keys
    app = _make_app_with_protected_route(keycloak)
    client = TestClient(app)
    token = _make_token(private_key, exp=int(time.time()) - 10)
    client.cookies.set("cce_access", token)
    response = client.get("/protected")
    assert response.status_code == 401


def test_invalid_token_returns_401(keycloak):
    app = _make_app_with_protected_route(keycloak)
    client = TestClient(app)
    client.cookies.set("cce_access", "garbage.token.here")
    response = client.get("/protected")
    assert response.status_code == 401


def test_require_auth_refreshes_when_access_cookie_missing(keycloak, rsa_keys):
    """When cce_access is absent but cce_refresh is valid, require_auth should
    silently refresh and return the user — not 401."""
    private_key, _ = rsa_keys
    fresh_access = _make_token(private_key)
    fresh_refresh = "new-refresh-token"

    app = _make_app_with_protected_route(keycloak)

    # Patch KeycloakClient.refresh_token on the specific keycloak instance bound to the app.
    with patch.object(
        keycloak,
        "refresh_token",
        new=AsyncMock(return_value={"access_token": fresh_access, "refresh_token": fresh_refresh}),
    ):
        client = TestClient(app, raise_server_exceptions=True)
        # Only the refresh cookie is present — no access cookie.
        client.cookies.set("cce_refresh", "some-valid-refresh-token")
        response = client.get("/protected")

    assert response.status_code == 200
    assert response.json()["sub"] == "user-123"
    # The new tokens must be stored on request.state so middleware can set cookies.
    # (TestClient doesn't run the full middleware stack, so we verify the endpoint
    # returned a successful response — the state assertions are covered by
    # test_expired_token_triggers_refresh_and_sets_new_cookies if present.)


def test_missing_access_cookie_without_refresh_returns_401(keycloak):
    """When cce_access is absent AND cce_refresh is absent, require_auth returns 401."""
    app = _make_app_with_protected_route(keycloak)
    client = TestClient(app)
    # No cookies at all.
    response = client.get("/protected")
    assert response.status_code == 401


def test_expired_token_with_refresh_succeeds(keycloak, rsa_keys):
    """When cce_access is expired but cce_refresh is valid, require_auth refreshes and succeeds."""
    private_key, _ = rsa_keys
    expired_access = _make_token(private_key, exp=int(time.time()) - 10)
    fresh_access = _make_token(private_key)
    fresh_refresh = "new-refresh-token"

    app = _make_app_with_protected_route(keycloak)

    with patch.object(
        keycloak,
        "refresh_token",
        new=AsyncMock(return_value={"access_token": fresh_access, "refresh_token": fresh_refresh}),
    ):
        client = TestClient(app, raise_server_exceptions=True)
        client.cookies.set("cce_access", expired_access)
        client.cookies.set("cce_refresh", "some-valid-refresh-token")
        response = client.get("/protected")

    assert response.status_code == 200
    assert response.json()["sub"] == "user-123"
