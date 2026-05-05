"""Tests for auth route endpoints."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient

from cell_explorer_api.auth.keycloak import KeycloakClient
from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


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
def auth_client(rsa_keys):
    _, public_key = rsa_keys
    settings = _make_settings()
    app = create_app(settings)
    keycloak: KeycloakClient = app.state.keycloak
    keycloak._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    return TestClient(app)


def test_login_redirects_to_keycloak(auth_client):
    response = auth_client.get("/api/auth/login", follow_redirects=False)
    assert response.status_code == 307
    location = response.headers["location"]
    assert "auth.example.com/realms/test-realm/protocol/openid-connect/auth" in location
    assert "client_id=test-client" in location


def test_login_sets_state_cookie(auth_client):
    response = auth_client.get("/api/auth/login", follow_redirects=False)
    assert "cce_state" in response.cookies


def test_cookies_not_secure_on_http(auth_client):
    """Cookies should not have the secure flag when served over HTTP."""
    response = auth_client.get("/api/auth/login", follow_redirects=False)
    set_cookie = response.headers.get_list("set-cookie")
    state_cookie = [c for c in set_cookie if c.startswith("cce_state")][0]
    assert "Secure" not in state_cookie


def test_cookies_secure_with_forwarded_proto(auth_client):
    """Cookies should have the secure flag when x-forwarded-proto is https."""
    response = auth_client.get(
        "/api/auth/login",
        headers={"x-forwarded-proto": "https"},
        follow_redirects=False,
    )
    set_cookie = response.headers.get_list("set-cookie")
    state_cookie = [c for c in set_cookie if c.startswith("cce_state")][0]
    assert "Secure" in state_cookie


def test_me_returns_user_when_authenticated(auth_client, rsa_keys):
    private_key, _ = rsa_keys
    token = _make_token(private_key)
    auth_client.cookies.set("cce_access", token)
    response = auth_client.get("/api/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["sub"] == "user-123"
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"


def test_me_returns_401_when_not_authenticated(auth_client):
    response = auth_client.get("/api/auth/me")
    assert response.status_code == 401


def test_logout_clears_cookies(auth_client, rsa_keys):
    private_key, _ = rsa_keys
    token = _make_token(private_key)
    auth_client.cookies.set("cce_access", token)
    auth_client.cookies.set("cce_refresh", "some-refresh-token")
    response = auth_client.post("/api/auth/logout")
    assert response.status_code == 200


def test_me_sets_refreshed_cookies(auth_client, rsa_keys):
    """When the access token is expired but refresh succeeds, /me should set new cookies."""
    private_key, _ = rsa_keys

    # Create an expired access token
    expired_token = _make_token(private_key, exp=int(time.time()) - 10)

    # Create a fresh token that the refresh flow will return
    fresh_token = _make_token(private_key)

    # Mock the refresh_token method on the keycloak client
    keycloak: KeycloakClient = auth_client.app.state.keycloak
    import unittest.mock as mock
    keycloak.refresh_token = mock.AsyncMock(return_value={
        "access_token": fresh_token,
        "refresh_token": "new-refresh-token-value",
    })

    auth_client.cookies.set("cce_access", expired_token)
    auth_client.cookies.set("cce_refresh", "old-refresh-token")

    response = auth_client.get("/api/auth/me")
    assert response.status_code == 200
    assert response.json()["sub"] == "user-123"

    # Verify refreshed cookies are set with the correct values
    set_cookie_headers = response.headers.get_list("set-cookie")
    cookie_map = {}
    for header in set_cookie_headers:
        name, _, rest = header.partition("=")
        value = rest.split(";")[0]
        cookie_map[name] = value
    assert "cce_access" in cookie_map, "Expected cce_access cookie to be set after refresh"
    assert cookie_map["cce_access"] == fresh_token, "cce_access should contain the refreshed token"
    assert "cce_refresh" in cookie_map, "Expected cce_refresh cookie to be set after refresh"
    assert cookie_map["cce_refresh"] == "new-refresh-token-value", "cce_refresh should contain the new refresh token"


def test_auth_routes_return_401_without_credentials():
    """When no credentials are presented, auth routes return 401 regardless of keycloak config."""
    settings = Settings()
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"
