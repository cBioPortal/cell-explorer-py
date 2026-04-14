"""Tests for dataset access (credential minting) endpoint."""

import time

import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.keycloak import KeycloakClient
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app


def _generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest_asyncio.fixture()
async def access_app(monkeypatch):
    """App with auth, database, and a private dataset."""
    monkeypatch.setenv("DATASOURCE_TEST_CDN_SIGNING_SECRET", "test-signing-secret")

    private_key, public_key = _generate_rsa_keypair()

    settings = Settings(
        database_url="sqlite+aiosqlite://",
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
    )
    app = create_app(settings)

    keycloak: KeycloakClient = app.state.keycloak
    keycloak._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="Test CDN",
            type=DatasourceType.HTTP_TOKEN,
            base_url="https://cdn.example.com",
            credential_ref="TEST_CDN",
        )
        session.add(ds)
        await session.flush()

        public_ds = Dataset(
            datasource_id=ds.id,
            name="Public Atlas",
            slug="public-atlas",
            path="datasets/public.zarr",
            is_public=True,
        )
        private_ds = Dataset(
            datasource_id=ds.id,
            name="Private Study",
            slug="private-study",
            path="datasets/private.zarr",
            is_public=False,
            required_roles=["lab-smith"],
        )
        session.add(public_ds)
        session.add(private_ds)
        await session.commit()

    app.state.db_engine = engine
    app.state.test_rsa_private_key = private_key
    return app


def _make_token(app, roles):
    private_key = app.state.test_rsa_private_key
    return jwt.encode(
        {
            "sub": "user-123",
            "name": "Test User",
            "realm_access": {"roles": roles},
            "iss": "https://auth.example.com/realms/test-realm",
            "aud": "test-client",
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )


def test_access_public_dataset_returns_url(access_app):
    client = TestClient(access_app)
    response = client.post("/api/datasets/public-atlas/access")
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == "https://cdn.example.com/datasets/public.zarr"
    assert data["credential_type"] == "public"


def test_access_private_dataset_requires_auth(access_app):
    client = TestClient(access_app)
    response = client.post("/api/datasets/private-study/access")
    assert response.status_code == 401


def test_access_private_dataset_requires_role(access_app):
    token = _make_token(access_app, roles=["other-role"])
    client = TestClient(access_app)
    client.cookies.set("cce_access", token)
    response = client.post("/api/datasets/private-study/access")
    assert response.status_code == 403


def test_access_private_dataset_with_role(access_app):
    token = _make_token(access_app, roles=["lab-smith"])
    client = TestClient(access_app)
    client.cookies.set("cce_access", token)
    response = client.post("/api/datasets/private-study/access")
    assert response.status_code == 200
    data = response.json()
    assert data["credential_type"] == "bearer_token"
    assert data["url"] == "https://cdn.example.com/datasets/private.zarr"
    assert "token" in data
    assert "expires_at" in data


def test_access_nonexistent_dataset(access_app):
    client = TestClient(access_app)
    response = client.post("/api/datasets/nonexistent/access")
    assert response.status_code == 404
