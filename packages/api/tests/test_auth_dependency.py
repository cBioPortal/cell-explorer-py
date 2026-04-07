"""Tests for require_auth dependency."""

import time

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
