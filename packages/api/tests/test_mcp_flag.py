"""Per-dataset MCP flag: private datasets are visible to MCP callers only when flagged."""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app
from cell_explorer_api.services.access import user_can_access

API_KEY = "test-admin-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


def _ds(*, is_public=False, roles=("r",), mcp_enabled=False):
    return SimpleNamespace(is_public=is_public, required_roles=list(roles), mcp_enabled=mcp_enabled)


def _user(roles=("r",), via_mcp=False):
    return User(sub="u1", roles=list(roles), via_mcp=via_mcp)


def test_mcp_caller_cannot_see_private_dataset_without_flag():
    assert user_can_access(_ds(), user=_user(via_mcp=True)) is False


def test_mcp_caller_sees_private_dataset_with_flag_and_role():
    assert user_can_access(_ds(mcp_enabled=True), user=_user(via_mcp=True)) is True


def test_mcp_flag_never_bypasses_roles():
    assert user_can_access(_ds(mcp_enabled=True), user=_user(roles=(), via_mcp=True)) is False


def test_public_dataset_ignores_flag_for_mcp_caller():
    assert user_can_access(_ds(is_public=True), user=_user(via_mcp=True)) is True


def test_website_caller_ignores_flag():
    assert user_can_access(_ds(), user=_user(via_mcp=False)) is True


@pytest_asyncio.fixture()
async def mcp_app():
    settings = Settings(
        database_url="sqlite+aiosqlite://",
        admin_api_key=API_KEY,
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
    )
    app = create_app(settings)
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        src = Datasource(
            name="CDN", type=DatasourceType.HTTP_TOKEN,
            base_url="https://cdn.example.com", credential_ref="TEST_CDN",
        )
        session.add(src)
        await session.flush()
        src_id = str(src.id)
        for slug, flag in (("hidden", False), ("exposed", True)):
            session.add(Dataset(
                datasource_id=src.id, name=slug, slug=slug, path=f"{slug}.zarr",
                is_public=False, required_roles=["r"], mcp_enabled=flag,
            ))
        await session.commit()
    app.state.db_engine = engine
    app.state.test_datasource_id = src_id
    return app


@pytest.fixture()
def client_with_mcp_user(mcp_app):
    mcp_app.dependency_overrides[optional_auth] = lambda: _user(via_mcp=True)
    return TestClient(mcp_app)


def test_hidden_dataset_is_404_for_mcp_caller(client_with_mcp_user):
    c = client_with_mcp_user
    assert c.get("/api/datasets/hidden").status_code == 404
    assert c.post("/api/datasets/hidden/access").status_code == 404
    slugs = [d["slug"] for d in c.get("/api/datasets").json()["datasets"]]
    assert "hidden" not in slugs
    assert "exposed" in slugs
    assert c.get("/api/datasets/exposed").status_code == 200


def test_hidden_dataset_still_visible_to_website_caller(mcp_app):
    mcp_app.dependency_overrides[optional_auth] = lambda: _user(via_mcp=False)
    c = TestClient(mcp_app)
    assert c.get("/api/datasets/hidden").status_code == 200


def test_admin_can_set_mcp_enabled(mcp_app):
    client = TestClient(mcp_app)
    created = client.post(
        "/api/admin/datasets", headers=AUTH_HEADER,
        json={"datasource_id": mcp_app.state.test_datasource_id, "name": "x", "slug": "x", "path": "x.zarr"},
    )
    assert created.status_code == 201, created.text
    assert created.json()["mcp_enabled"] is False
    response = client.put("/api/admin/datasets/x", headers=AUTH_HEADER, json={"mcp_enabled": True})
    assert response.status_code == 200, response.text
    assert response.json()["mcp_enabled"] is True
