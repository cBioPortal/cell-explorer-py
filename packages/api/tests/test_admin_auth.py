"""Tests for admin authentication dependency."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from cell_explorer_api.auth.admin import require_admin
from cell_explorer_api.auth.keycloak import KeycloakClient
from cell_explorer_api.config import Settings


def _make_test_app(settings: Settings) -> FastAPI:
    """Create a minimal app with one admin-protected route."""
    from cell_explorer_api.main import create_app

    app = create_app(settings)
    admin_router = APIRouter(prefix="/api/admin", tags=["admin"])

    @admin_router.get("/test")
    async def admin_test(admin=Depends(require_admin)):
        return {"ok": True}

    app.include_router(admin_router)
    return app


def test_admin_returns_501_when_not_configured():
    settings = Settings()
    app = _make_test_app(settings)
    client = TestClient(app)
    response = client.get("/api/admin/test")
    assert response.status_code == 501


def test_admin_accepts_valid_api_key():
    settings = Settings(admin_api_key="test-secret-key")
    app = _make_test_app(settings)
    client = TestClient(app)
    response = client.get(
        "/api/admin/test",
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert response.status_code == 200


def test_admin_rejects_invalid_api_key():
    settings = Settings(admin_api_key="test-secret-key")
    app = _make_test_app(settings)
    client = TestClient(app)
    response = client.get(
        "/api/admin/test",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 403


def test_admin_rejects_missing_header():
    settings = Settings(admin_api_key="test-secret-key")
    app = _make_test_app(settings)
    client = TestClient(app)
    response = client.get("/api/admin/test")
    assert response.status_code == 403


def _generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def test_admin_accepts_keycloak_admin_role():
    """When Keycloak is enabled, a JWT with 'admin' role grants admin access."""
    private_key, public_key = _generate_rsa_keypair()
    settings = Settings(
        admin_api_key="test-key",
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
    )
    app = _make_test_app(settings)
    keycloak: KeycloakClient = app.state.keycloak
    keycloak._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }

    token = jwt.encode(
        {
            "sub": "admin-user",
            "name": "Admin",
            "realm_access": {"roles": ["admin"]},
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
    response = client.get("/api/admin/test")
    assert response.status_code == 200
