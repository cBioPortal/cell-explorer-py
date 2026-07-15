"""Tests for dataset discovery endpoints."""

import time
import uuid

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


@pytest.fixture()
def db_url():
    return "sqlite+aiosqlite://"


@pytest.fixture()
def settings(db_url):
    return Settings(database_url=db_url)


@pytest.fixture()
def app(settings):
    return create_app(settings)


@pytest_asyncio.fixture()
async def seeded_app(app, db_url):
    """Create tables and seed test data."""
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="Test CDN",
            type=DatasourceType.S3_CLOUDFRONT,
            base_url="https://cdn.example.com",
        )
        session.add(ds)
        await session.flush()

        public_dataset = Dataset(
            datasource_id=ds.id,
            name="Public Atlas",
            slug="public-atlas",
            path="datasets/public.zarr",
            is_public=True,
            chat_enabled=True,
            default_view={"pointSize": 3},
        )
        private_dataset = Dataset(
            datasource_id=ds.id,
            name="Private Study",
            slug="private-study",
            path="datasets/private.zarr",
            is_public=False,
            required_roles=["lab-smith"],
        )
        session.add(public_dataset)
        session.add(private_dataset)
        await session.commit()

    # Point the app's engine to the seeded database
    app.state.db_engine = engine
    return app


def test_list_datasets_anonymous_sees_public_only(seeded_app):
    client = TestClient(seeded_app)
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert len(data["datasets"]) == 1
    assert data["datasets"][0]["slug"] == "public-atlas"
    assert data["datasets"][0]["url"] == "https://cdn.example.com/datasets/public.zarr"


def test_list_datasets_authenticated_sees_authorized(seeded_app):
    private_key, public_key = _generate_rsa_keypair()
    settings = seeded_app.state.settings
    # Enable Keycloak for this test
    settings.keycloak_url = "https://auth.example.com"
    settings.keycloak_realm = "test-realm"
    settings.keycloak_client_id = "test-client"
    settings.keycloak_client_secret = "test-secret"

    keycloak = KeycloakClient(settings)
    keycloak._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    seeded_app.state.keycloak = keycloak

    token = jwt.encode(
        {
            "sub": "user-123",
            "name": "Test User",
            "realm_access": {"roles": ["lab-smith"]},
            "iss": "https://auth.example.com/realms/test-realm",
            "aud": "test-client",
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )
    client = TestClient(seeded_app)
    client.cookies.set("cce_access", token)
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    slugs = [d["slug"] for d in data["datasets"]]
    assert "public-atlas" in slugs
    assert "private-study" in slugs


def test_list_datasets_authenticated_no_matching_role(seeded_app):
    private_key, public_key = _generate_rsa_keypair()
    settings = seeded_app.state.settings
    settings.keycloak_url = "https://auth.example.com"
    settings.keycloak_realm = "test-realm"
    settings.keycloak_client_id = "test-client"
    settings.keycloak_client_secret = "test-secret"

    keycloak = KeycloakClient(settings)
    keycloak._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    seeded_app.state.keycloak = keycloak

    token = jwt.encode(
        {
            "sub": "user-456",
            "name": "Other User",
            "realm_access": {"roles": ["other-role"]},
            "iss": "https://auth.example.com/realms/test-realm",
            "aud": "test-client",
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )
    client = TestClient(seeded_app)
    client.cookies.set("cce_access", token)
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert len(data["datasets"]) == 1
    assert data["datasets"][0]["slug"] == "public-atlas"


def test_get_dataset_by_slug(seeded_app):
    client = TestClient(seeded_app)
    response = client.get("/api/datasets/public-atlas")
    assert response.status_code == 200
    data = response.json()
    assert data["slug"] == "public-atlas"
    assert data["url"] == "https://cdn.example.com/datasets/public.zarr"


def test_get_dataset_not_found(seeded_app):
    client = TestClient(seeded_app)
    response = client.get("/api/datasets/nonexistent")
    assert response.status_code == 404


def test_private_dataset_url_is_null_for_anonymous(seeded_app):
    """Anonymous users who somehow request a private dataset get url=null."""
    private_key, public_key = _generate_rsa_keypair()
    settings = seeded_app.state.settings
    settings.keycloak_url = "https://auth.example.com"
    settings.keycloak_realm = "test-realm"
    settings.keycloak_client_id = "test-client"
    settings.keycloak_client_secret = "test-secret"

    keycloak = KeycloakClient(settings)
    keycloak._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    seeded_app.state.keycloak = keycloak

    # Authenticated user with matching role
    token = jwt.encode(
        {
            "sub": "user-123",
            "realm_access": {"roles": ["lab-smith"]},
            "iss": "https://auth.example.com/realms/test-realm",
            "aud": "test-client",
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )
    client = TestClient(seeded_app)
    client.cookies.set("cce_access", token)
    response = client.get("/api/datasets/private-study")
    assert response.status_code == 200
    data = response.json()
    assert data["url"] is None  # Must call /access to get credentials


def test_list_datasets_includes_chat_enabled(seeded_app):
    """The list endpoint surfaces the per-dataset chat opt-in."""
    client = TestClient(seeded_app)
    response = client.get("/api/datasets")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert len(datasets) > 0
    # Every entry has chat_enabled populated
    for d in datasets:
        assert "chat_enabled" in d
        assert isinstance(d["chat_enabled"], bool)


def test_get_dataset_includes_chat_enabled(seeded_app):
    """The detail endpoint also surfaces chat_enabled."""
    client = TestClient(seeded_app)
    response = client.get("/api/datasets/public-atlas")
    assert response.status_code == 200
    assert response.json()["chat_enabled"] is True  # seed has chat_enabled=True


def test_get_dataset_includes_default_view(seeded_app):
    client = TestClient(seeded_app)
    resp = client.get("/api/datasets/public-atlas")
    assert resp.status_code == 200
    assert resp.json()["default_view"] == {"pointSize": 3}
