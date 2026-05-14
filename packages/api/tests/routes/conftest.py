"""Shared fixtures + helpers for routes/* test modules.

Anything used by more than one test file in this directory lives here.
Test-file-specific seeders (e.g. fake agents, single-test data setup)
stay in their respective test modules.
"""

import time

import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient
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
    return Settings(database_url=db_url, anthropic_api_key="sk-ant-test")


@pytest.fixture()
def app(settings):
    return create_app(settings)


@pytest_asyncio.fixture()
async def seeded_app(app, db_url):
    from cell_explorer_api.db import create_engine
    # Use project helper so the SQLite foreign_keys pragma listener is
    # attached automatically (needed for ON DELETE CASCADE in tests).
    engine = create_engine(db_url)
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
            chat_enabled=True,
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
