"""Admin endpoints: register MCP servers, end a user's MCP connections."""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app
from cell_explorer_api.services import oauth_store as store

BASE = "https://ce.test"
RES = "https://mcp.test/mcp"
ADMIN = {"Authorization": "Bearer admin-key"}
EXCHANGE = "/api/oauth/token-exchange"
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
async def client(engine):
    app = create_app(
        Settings(
            database_url="sqlite+aiosqlite://",
            admin_api_key="admin-key",
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


async def _register(client, url=RES, name="MCP"):
    resp = await client.post("/api/admin/mcp-resources", json={"url": url, "name": name}, headers=ADMIN)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def test_create_mcp_resource_returns_key_once(client, engine):
    body = await _register(client)
    assert set(body) == {"id", "url", "name", "key"}
    assert body["url"] == RES and body["key"]
    async with AsyncSession(engine) as db:
        found = await store.find_mcp_resource_by_key(db, body["key"])
    assert found is not None and str(found.id) == body["id"]
    dup = await client.post("/api/admin/mcp-resources", json={"url": RES, "name": "x"}, headers=ADMIN)
    assert dup.status_code == 409
    assert body["key"] not in dup.text


async def test_create_validates_url_and_requires_admin(client):
    for bad in ("http://mcp.test/mcp", "ftp://x.test", "not a url", "https://"):
        resp = await client.post("/api/admin/mcp-resources", json={"url": bad, "name": "x"}, headers=ADMIN)
        assert resp.status_code == 422, bad
    for ok in ("http://localhost:8080/mcp", "http://127.0.0.1/mcp"):
        resp = await client.post("/api/admin/mcp-resources", json={"url": ok, "name": "x"}, headers=ADMIN)
        assert resp.status_code == 201, ok
    resp = await client.post("/api/admin/mcp-resources", json={"url": RES, "name": "x"})
    assert resp.status_code in (401, 403)


async def test_list_never_shows_key(client):
    body = await _register(client)
    resp = await client.get("/api/admin/mcp-resources", headers=ADMIN)
    assert resp.status_code == 200
    [row] = resp.json()["mcp_resources"]
    assert set(row) == {"id", "url", "name", "created_at", "revoked_at"}
    assert row["revoked_at"] is None
    assert body["key"] not in resp.text


async def test_revoked_resource_key_cannot_exchange(client, engine):
    body = await _register(client)
    async with AsyncSession(engine) as db:
        grant = await store.create_grant(
            db, user=User(sub="u1", email="u@x.org", name="U", roles=[]), client_id="c1", resource=RES
        )
        access, _ = await store.issue_token(db, grant=grant, kind="access", audience=RES)
    form = {"grant_type": GRANT_TYPE, "subject_token": access, "subject_token_type": TOKEN_TYPE}
    key_header = {"Authorization": f"Bearer {body['key']}"}
    assert (await client.post(EXCHANGE, data=form, headers=key_header)).status_code == 200
    resp = await client.delete(f"/api/admin/mcp-resources/{body['id']}", headers=ADMIN)
    assert resp.status_code == 204
    resp = await client.post(EXCHANGE, data=form, headers=key_header)
    assert resp.status_code == 401
    assert resp.json() == {"error": "invalid_client"}
    [row] = (await client.get("/api/admin/mcp-resources", headers=ADMIN)).json()["mcp_resources"]
    assert row["revoked_at"] is not None


async def test_revoke_unknown_resource_404(client):
    resp = await client.delete("/api/admin/mcp-resources/00000000-0000-0000-0000-000000000000", headers=ADMIN)
    assert resp.status_code == 404


async def test_admin_revoke_invalidates_issued_tokens(client, engine):
    body = await _register(client)
    async with AsyncSession(engine) as db:
        grant = await store.create_grant(
            db, user=User(sub="u1", email="u@x.org", name="U", roles=["admin"]), client_id="c1", resource=RES
        )
        access, _ = await store.issue_token(db, grant=grant, kind="access", audience=RES)
        api, _ = await store.issue_token(db, grant=grant, kind="api", audience="api")
        other = User(sub="u2", email="o@x.org", name="O", roles=[])
        other_grant = await store.create_grant(db, user=other, client_id="c1", resource=RES)
        other_access, _ = await store.issue_token(db, grant=other_grant, kind="access", audience=RES)
    key_header = {"Authorization": f"Bearer {body['key']}"}

    def form(t):
        return {"grant_type": GRANT_TYPE, "subject_token": t, "subject_token_type": TOKEN_TYPE}

    # API tokens never satisfy admin auth.
    resp = await client.get("/api/admin/mcp-resources", headers={"Authorization": f"Bearer {api}"})
    assert resp.status_code in (401, 403)

    resp = await client.delete("/api/admin/oauth/grants/u1", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.json() == {"revoked": 1}
    async with AsyncSession(engine) as db:
        assert await store.find_token(db, api, kind="api") is None
        assert await store.find_token(db, access, kind="access") is None
    resp = await client.post(EXCHANGE, data=form(access), headers=key_header)
    assert resp.status_code == 400 and resp.json() == {"error": "invalid_grant"}
    # Other users unaffected; repeat revoke is a no-op count.
    assert (await client.post(EXCHANGE, data=form(other_access), headers=key_header)).status_code == 200
    resp = await client.delete("/api/admin/oauth/grants/u1", headers=ADMIN)
    assert resp.json() == {"revoked": 0}
    # The revoked user's API token is now anonymous on dataset routes.
    resp = await client.get("/api/datasets/nope", headers={"Authorization": f"Bearer {api}"})
    assert resp.status_code in (401, 404)
    resp = await client.post("/api/datasets/nope/access", headers={"Authorization": f"Bearer {api}"})
    assert resp.status_code in (401, 404)
