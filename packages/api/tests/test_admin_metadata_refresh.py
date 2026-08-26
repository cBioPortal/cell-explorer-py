"""Admin metadata refresh endpoints: single-dataset and bulk refresh.

Async, driven with `httpx.AsyncClient` + `ASGITransport`, pinned to the same
session-scoped event loop as the `zarr_server` fixture — see
test_admin_dataset_metadata.py's module docstring for why a plain sync
`fastapi.testclient.TestClient` does not work here: fsspec caches its async
filesystem instances process-globally, and a sync `TestClient` opens a fresh
event loop per request when not used as a context manager, so a later call
against the same cached filesystem instance dies with "Event loop is closed".
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_KEY = "test-admin-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture()
def app():
    return create_app(Settings(database_url="sqlite+aiosqlite://", admin_api_key=API_KEY))


@pytest_asyncio.fixture(loop_scope="session")
async def seeded_app(app, zarr_server):
    """Two datasets: one readable, one pointing nowhere. Neither harvested yet."""
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    # expire_on_commit=False: without it, accessing attributes after commit()
    # below triggers an implicit lazy-load refresh, which raises MissingGreenlet
    # under the async driver (see test_metadata_harvest.py's _engine_with_dataset
    # for the same pattern in this codebase).
    async with AsyncSession(engine, expire_on_commit=False) as session:
        ds = Datasource(
            name="Fixture server",
            type=DatasourceType.S3_CLOUDFRONT,
            base_url=zarr_server,
        )
        session.add(ds)
        await session.flush()
        session.add(
            Dataset(
                datasource_id=ds.id,
                name="Good",
                slug="good",
                path="tiny_v3.zarr",
                is_public=True,
            )
        )
        session.add(
            Dataset(
                datasource_id=ds.id,
                name="Broken",
                slug="broken",
                path="does_not_exist.zarr",
                is_public=True,
            )
        )
        await session.commit()
    app.state.db_engine = engine
    return app


async def test_refresh_single_dataset(seeded_app):
    async with AsyncClient(transport=ASGITransport(app=seeded_app), base_url="http://test") as client:
        res = await client.post("/api/admin/datasets/good/metadata/refresh", headers=AUTH_HEADER)
    assert res.status_code == 200
    assert res.json()["metadata"]["n_obs"] == 12


async def test_refresh_single_unknown_slug_returns_404(seeded_app):
    async with AsyncClient(transport=ASGITransport(app=seeded_app), base_url="http://test") as client:
        res = await client.post("/api/admin/datasets/nope/metadata/refresh", headers=AUTH_HEADER)
    assert res.status_code == 404


async def test_refresh_single_records_error_without_failing(seeded_app):
    async with AsyncClient(transport=ASGITransport(app=seeded_app), base_url="http://test") as client:
        res = await client.post("/api/admin/datasets/broken/metadata/refresh", headers=AUTH_HEADER)
    assert res.status_code == 200, "a failed harvest must not fail the request"
    assert res.json()["metadata"]["status"] == "error"


async def test_bulk_refresh_reports_each_dataset(seeded_app):
    async with AsyncClient(transport=ASGITransport(app=seeded_app), base_url="http://test") as client:
        res = await client.post(
            "/api/admin/datasets/metadata/refresh",
            json={"only_stale": False},
            headers=AUTH_HEADER,
        )
    assert res.status_code == 200
    body = res.json()
    assert body["refreshed"] == 2
    by_slug = {r["slug"]: r for r in body["results"]}
    assert by_slug["good"]["status"] == "ok"
    assert by_slug["broken"]["status"] == "error"
    assert by_slug["broken"]["error"]


async def test_bulk_refresh_only_stale_skips_fresh(seeded_app):
    async with AsyncClient(transport=ASGITransport(app=seeded_app), base_url="http://test") as client:
        await client.post("/api/admin/datasets/good/metadata/refresh", headers=AUTH_HEADER)

        res = await client.post(
            "/api/admin/datasets/metadata/refresh",
            json={"only_stale": True},
            headers=AUTH_HEADER,
        )
    assert res.status_code == 200
    slugs = {r["slug"] for r in res.json()["results"]}
    assert "good" not in slugs, "a just-harvested dataset is not stale"
    assert "broken" in slugs, "never-succeeded datasets are always stale"


async def test_refresh_requires_admin_auth(seeded_app):
    # This app's require_admin dependency raises 403 (not 401) for a missing
    # Authorization header — see auth/admin.py and
    # test_admin_routes.py::test_admin_requires_auth for the same contract.
    async with AsyncClient(transport=ASGITransport(app=seeded_app), base_url="http://test") as client:
        res = await client.post("/api/admin/datasets/good/metadata/refresh")
    assert res.status_code == 403
