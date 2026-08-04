"""Tests for optional auth dependency."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from cell_explorer_api.auth.oidc import OidcClient
from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.config import Settings


def _generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _make_test_app(settings: Settings) -> FastAPI:
    from cell_explorer_api.main import create_app

    app = create_app(settings)
    test_router = APIRouter(prefix="/api/test")

    @test_router.get("/whoami")
    async def whoami(user: User | None = Depends(optional_auth)):
        if user is None:
            return {"authenticated": False}
        return {"authenticated": True, "sub": user.sub}

    app.include_router(test_router)
    return app


def test_optional_auth_returns_none_when_anonymous():
    settings = Settings()
    app = _make_test_app(settings)
    client = TestClient(app)
    response = client.get("/api/test/whoami")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False}


def test_optional_auth_returns_user_when_authenticated():
    private_key, public_key = _generate_rsa_keypair()
    settings = Settings(
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
    )
    app = _make_test_app(settings)
    oidc: OidcClient = app.state.oidc
    oidc._apply_discovery({
        "issuer": "https://auth.example.com/realms/test-realm",
        "authorization_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/auth",
        "token_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/token",
        "jwks_uri": "https://auth.example.com/realms/test-realm/protocol/openid-connect/certs",
        "end_session_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/logout",
    })
    oidc._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    token = jwt.encode(
        {
            "sub": "user-123",
            "name": "Test User",
            "realm_access": {"roles": ["viewer"]},
            "iss": "https://auth.example.com/realms/test-realm",
            "aud": "test-client",
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )
    client = TestClient(app)
    client.cookies.set("cce_access", token)
    response = client.get("/api/test/whoami")
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["sub"] == "user-123"


def test_optional_auth_returns_none_with_invalid_token():
    settings = Settings(
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
    )
    app = _make_test_app(settings)
    client = TestClient(app)
    client.cookies.set("cce_access", "invalid-token")
    response = client.get("/api/test/whoami")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False}
