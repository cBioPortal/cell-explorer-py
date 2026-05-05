"""Tests for /api/chat/{slug} routes."""

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

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
        session.add(Dataset(
            datasource_id=ds.id,
            name="Public Atlas",
            slug="public-atlas",
            path="datasets/public.zarr",
            is_public=True,
        ))
        await session.commit()

    app.state.db_engine = engine
    return app


def _set_auth_cookie(client: TestClient, app, *, sub="user-1", roles=None):
    """Generate an RSA-signed test JWT and set it as the cce_access cookie."""
    private_key, public_key = _generate_rsa_keypair()
    settings = app.state.settings
    settings.keycloak_url = "https://auth.example.com"
    settings.keycloak_realm = "test-realm"
    settings.keycloak_client_id = "test-client"
    settings.keycloak_client_secret = "test-secret"
    keycloak = KeycloakClient(settings)
    keycloak._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    app.state.keycloak = keycloak

    token = jwt.encode(
        {
            "sub": sub,
            "name": "Test User",
            "realm_access": {"roles": list(roles or [])},
            "iss": "https://auth.example.com/realms/test-realm",
            "aud": "test-client",
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )
    client.cookies.set("cce_access", token)


# ---------- fake agent shared across tests ----------

@dataclass
class _FakeObsCol:
    name: str
    dtype: str = "categorical"
    cardinality: int | None = 5


@dataclass
class _FakeDatasetCtx:
    slug: str = "public-atlas"
    name: str = "Public Atlas"
    description: str = "A public test atlas"
    n_obs: int = 100
    n_var: int = 50
    obs_columns: list = None  # set in __post_init__
    embedding_keys: list = None

    def __post_init__(self):
        if self.obs_columns is None:
            self.obs_columns = [_FakeObsCol(name="cell_type")]
        if self.embedding_keys is None:
            self.embedding_keys = ["X_umap"]


@dataclass
class _FakeTool:
    name: str


@dataclass
class _FakeCatalog:
    tools: list

    def all(self):
        return self.tools


class _FakeChatAgent:
    """Minimal stand-in for ChatAgent. Tests script .run() per case."""

    def __init__(self, *, ctx=None, catalog=None, run_events=None, run_raises=None):
        self.dataset_ctx = ctx or _FakeDatasetCtx()
        self.catalog = catalog or _FakeCatalog([_FakeTool("get_dataset_schema")])
        self._run_events = run_events or []
        self._run_raises = run_raises  # exception or callable[int -> exception | None]

    async def run(self, *, messages):
        for i, ev in enumerate(self._run_events):
            yield ev
            if self._run_raises is not None:
                exc = self._run_raises(i) if callable(self._run_raises) else self._run_raises
                if exc:
                    raise exc


# ---------- GET /context tests ----------

def test_get_context_happy_path(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["slug"] == "public-atlas"
    assert data["name"] == "Public Atlas"
    assert data["description"] == "A public test atlas"
    assert data["n_obs"] == 100
    assert data["n_var"] == 50
    assert data["obs_columns"] == [{"name": "cell_type", "dtype": "categorical", "cardinality": 5}]
    assert data["embedding_keys"] == ["X_umap"]
    assert data["available_tools"] == ["get_dataset_schema"]


def test_get_context_unauthenticated_returns_401(seeded_app):
    client = TestClient(seeded_app)
    # No cookie set.
    response = client.get("/api/chat/public-atlas/context")
    assert response.status_code == 401


def test_get_context_dataset_not_found_returns_404(seeded_app):
    from cell_explorer_api.services.chat_session import DatasetNotFoundError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise DatasetNotFoundError("dataset 'nope' not found")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/nope/context")

    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_get_context_access_denied_returns_403(seeded_app):
    from cell_explorer_api.services.chat_session import AccessDeniedError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise AccessDeniedError("access denied; required roles: ['admin']")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 403
    assert "admin" in response.json()["detail"]


def test_get_context_credential_mint_error_returns_502(seeded_app):
    from cell_explorer_api.services.chat_session import CredentialMintError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise CredentialMintError("signing key missing")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 502
    assert "credentials" in response.json()["detail"].lower()


def test_get_context_zarr_unreachable_returns_502(seeded_app):
    from cell_explorer_api.services.chat_session import ZarrUnreachableError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise ZarrUnreachableError("connection refused")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 502
    assert "zarr" in response.json()["detail"].lower()
