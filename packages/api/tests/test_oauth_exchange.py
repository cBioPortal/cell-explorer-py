"""Token exchange: an MCP server trades its key + a user's access token for an API token."""

from datetime import timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import utcnow
from cell_explorer_api.db.oauth_models import OAuthGrant, OAuthToken
from cell_explorer_api.main import create_app
from cell_explorer_api.services import oauth_store as store

BASE = "https://ce.test"
RES = "https://mcp.test/mcp"
OTHER_RES = "https://other.test/mcp"
URL = "/api/oauth/token-exchange"
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def world(engine):
    """An MCP resource with its key, plus a grant with an access token for it."""
    async with AsyncSession(engine) as db:
        key, _ = await store.create_mcp_resource(db, url=RES, name="MCP")
        user = User(sub="user-1", name="Ada", email="ada@example.org", roles=["lab"])
        grant = await store.create_grant(db, user=user, client_id="c1", resource=RES)
        access, _ = await store.issue_token(db, grant=grant, kind="access", audience=RES)
        refresh, _ = await store.issue_token(db, grant=grant, kind="refresh", audience=RES)
        api, _ = await store.issue_token(db, grant=grant, kind="api", audience="api")
        other, _ = await store.issue_token(db, grant=grant, kind="access", audience=OTHER_RES)
        return dict(key=key, grant=grant, access=access, refresh=refresh, api=api, other=other)


@pytest_asyncio.fixture
async def client(engine, world):
    app = create_app(
        Settings(
            database_url="sqlite+aiosqlite://",
            keycloak_url="https://auth.example.com",
            keycloak_realm="test-realm",
            keycloak_client_id="test-client",
            keycloak_client_secret="test-secret",
            public_url=BASE,
        )
    )
    await app.state.db_engine.dispose()
    app.state.db_engine = engine
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        yield c


def _form(token, **extra):
    return {"grant_type": GRANT_TYPE, "subject_token": token, "subject_token_type": TOKEN_TYPE, **extra}


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


async def test_exchange_returns_api_token(client, engine, world):
    resp = await client.post(URL, data=_form(world["access"]), headers=_auth(world["key"]))
    assert resp.status_code == 200, resp.text
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["issued_token_type"] == TOKEN_TYPE
    assert body["token_type"] == "Bearer"
    assert body["expires_in"] == 600
    assert body["subject"] == "user-1"
    async with AsyncSession(engine) as db:
        found = await store.find_token(db, body["access_token"], kind="api")
    assert found is not None
    token, grant = found
    assert token.audience == "api"
    assert grant.id == world["grant"].id


async def test_exchange_requires_mcp_key(client, engine, world):
    async with AsyncSession(engine) as db:
        resource = await store.find_mcp_resource_by_key(db, world["key"])
        resource.revoked_at = utcnow()
        db.add(resource)
        await db.commit()
    for headers in ({}, _auth("wrong"), {"Authorization": "Basic abc"}, _auth(world["key"])):
        resp = await client.post(URL, data=_form(world["access"]), headers=headers)
        assert resp.status_code == 401, headers
        assert resp.json() == {"error": "invalid_client"}


async def test_exchange_refuses_refresh_and_api_tokens(client, world):
    for name in ("refresh", "api", "nonsense"):
        token = world.get(name, name)
        resp = await client.post(URL, data=_form(token), headers=_auth(world["key"]))
        assert resp.status_code == 400, name
        assert resp.json() == {"error": "invalid_grant"}
    resp = await client.post(
        URL, data={"grant_type": GRANT_TYPE, "subject_token_type": TOKEN_TYPE}, headers=_auth(world["key"])
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid_grant"}


async def test_exchange_refuses_token_for_other_resource(client, world):
    resp = await client.post(URL, data=_form(world["other"]), headers=_auth(world["key"]))
    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid_grant"}


async def test_exchange_refuses_revoked_grant(client, engine, world):
    async with AsyncSession(engine) as db:
        await store.revoke_grant(db, world["grant"].id)
    resp = await client.post(URL, data=_form(world["access"]), headers=_auth(world["key"]))
    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid_grant"}


async def test_exchange_refuses_other_grant_type(client, world):
    resp = await client.post(
        URL, data={**_form(world["access"]), "grant_type": "client_credentials"}, headers=_auth(world["key"])
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": "unsupported_grant_type"}


@pytest.mark.parametrize(
    "token_type", [None, "urn:ietf:params:oauth:token-type:refresh_token", "urn:ietf:params:oauth:token-type:jwt"]
)
async def test_exchange_refuses_other_subject_token_type(client, world, token_type):
    form = _form(world["access"])
    if token_type is None:
        del form["subject_token_type"]
    else:
        form["subject_token_type"] = token_type
    resp = await client.post(URL, data=form, headers=_auth(world["key"]))
    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid_request"}


async def test_exchange_checks_key_before_reading_body(client, world, monkeypatch):
    from starlette.requests import Request

    async def no_form(self, *a, **k):
        raise AssertionError("body read before the MCP key was checked")

    monkeypatch.setattr(Request, "form", no_form)
    for headers in ({}, _auth("wrong")):
        resp = await client.post(URL, data=_form(world["access"]), headers=headers)
        assert resp.status_code == 401
        assert resp.json() == {"error": "invalid_client"}


async def _set_expiry(engine, model, key, seconds: float):
    async with AsyncSession(engine) as db:
        row = await db.get(model, key)
        row.expires_at = utcnow() + timedelta(seconds=seconds)
        db.add(row)
        await db.commit()


async def test_api_token_capped_at_access_token_expiry(client, engine, world):
    await _set_expiry(engine, OAuthToken, store.hash_secret(world["access"]), 100)
    resp = await client.post(URL, data=_form(world["access"]), headers=_auth(world["key"]))
    assert resp.status_code == 200, resp.text
    assert 95 <= resp.json()["expires_in"] <= 100
    async with AsyncSession(engine) as db:
        api = await db.get(OAuthToken, store.hash_secret(resp.json()["access_token"]))
        access = await db.get(OAuthToken, store.hash_secret(world["access"]))
    assert api.expires_at <= access.expires_at


async def test_expires_in_follows_grant_cap(client, engine, world):
    await _set_expiry(engine, OAuthGrant, world["grant"].id, 30)
    resp = await client.post(URL, data=_form(world["access"]), headers=_auth(world["key"]))
    assert resp.status_code == 200, resp.text
    assert 1 <= resp.json()["expires_in"] <= 30
