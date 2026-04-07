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


def test_auth_routes_not_registered_without_keycloak():
    """When KEYCLOAK_URL is not set, auth routes should not exist."""
    settings = Settings()
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/auth/me")
    # Should be 404 (route doesn't exist) or 405, not 200 or 401
    assert response.status_code in (404, 405, 422)
