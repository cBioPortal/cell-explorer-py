"""API tokens (Authorization: Bearer) on the dataset routes."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app
from cell_explorer_api.services import oauth_store as store

BASE = "https://ce.test"
RES = "https://mcp.test/mcp"
ADMIN_KEY = "admin-key"


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
async def world(engine, monkeypatch):
    monkeypatch.setattr(
        "cell_explorer_api.routes.datasets.mint_credentials",
        lambda ds, path: {"credential_type": "token", "url": "https://cdn.example.com/x", "token": "t"},
    )
    async with AsyncSession(engine) as db:
        src = Datasource(
            name="CDN", type=DatasourceType.HTTP_TOKEN,
            base_url="https://cdn.example.com", credential_ref="TEST_CDN",
        )
        db.add(src)
        await db.flush()
        for slug, flag in (("hidden", False), ("exposed", True)):
            db.add(Dataset(
                datasource_id=src.id, name=slug, slug=slug, path=f"{slug}.zarr",
                is_public=False, required_roles=["r"], mcp_enabled=flag,
            ))
        await db.commit()
        user = User(sub="u1", email="u@example.org", name="U", roles=["r", "admin"])
        grant = await store.create_grant(db, user=user, client_id="c1", resource=RES)
        tokens = {}
        for kind, aud in (("api", "api"), ("access", RES), ("refresh", RES)):
            tokens[kind], _ = await store.issue_token(db, grant=grant, kind=kind, audience=aud)
        return tokens


@pytest_asyncio.fixture
async def app(engine, world):
    app = create_app(Settings(
        database_url="sqlite+aiosqlite://",
        admin_api_key=ADMIN_KEY,
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
        public_url=BASE,
    ))
    await app.state.db_engine.dispose()
    app.state.db_engine = engine
    return app


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        yield c


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


async def _slugs(client, headers=None):
    resp = await client.get("/api/datasets", headers=headers)
    assert resp.status_code == 200
    return [d["slug"] for d in resp.json()["datasets"]]


async def test_api_token_sees_flagged_private_dataset(client, world):
    h = _bearer(world["api"])
    assert await _slugs(client, h) == ["exposed"]
    assert (await client.get("/api/datasets/exposed", headers=h)).status_code == 200
    assert (await client.post("/api/datasets/exposed/access", headers=h)).status_code == 200


async def test_unflagged_private_dataset_404_for_api_token(client, world):
    h = _bearer(world["api"])
    assert (await client.get("/api/datasets/hidden", headers=h)).status_code == 404
    assert (await client.post("/api/datasets/hidden/access", headers=h)).status_code == 404


async def test_access_and_refresh_tokens_are_anonymous_on_api(client, world):
    for name in ("access", "refresh"):
        assert await _slugs(client, _bearer(world[name])) == [], name
        assert (await client.get("/api/datasets/exposed", headers=_bearer(world[name]))).status_code == 401
    assert await _slugs(client, _bearer("nonsense")) == []


async def test_cookie_wins_over_bearer(app, client, world):
    app.dependency_overrides[optional_auth] = lambda: User(sub="w", roles=["r"])
    assert sorted(await _slugs(client, _bearer(world["api"]))) == ["exposed", "hidden"]


async def test_bearer_call_sets_no_cookies(client, world):
    resp = await client.get("/api/datasets", headers=_bearer(world["api"]))
    assert "set-cookie" not in resp.headers
    resp = await client.post("/api/datasets/exposed/access", headers=_bearer(world["api"]))
    assert "set-cookie" not in resp.headers


async def test_api_token_is_not_admin(client, world):
    resp = await client.get("/api/admin/datasets", headers=_bearer(world["api"]))
    assert resp.status_code == 403
