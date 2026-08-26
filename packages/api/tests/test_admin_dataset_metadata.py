"""Harvest-on-write behaviour for admin dataset endpoints.

These tests are async and driven with `httpx.AsyncClient` + `ASGITransport`
rather than the synchronous `fastapi.testclient.TestClient`, for two reasons
found while writing this suite:

1. `zarr_server` (conftest.py) is a pytest-asyncio fixture pinned to a
   session-scoped event loop (`loop_scope="session"`) — see
   test_metadata_harvest.py's `pytestmark` for why. A plain sync test has no
   pytest-asyncio-managed loop of its own to resolve that fixture against.

2. Even serving the fixture stores over a plain synchronous threaded HTTP
   server sidesteps (1) but not a second problem: `TestClient(app)` used
   without a `with` block opens a *fresh* anyio portal — and therefore a
   fresh event loop — per request (see `starlette.testclient`'s
   `_portal_factory`), and even wrapping a test's several calls in one
   `with TestClient(app) as client:` block only fixes it *within* that test.
   zarr-access's store opening goes through fsspec's `FsspecStore`, and
   fsspec caches its async filesystem instances at the process level, keyed
   by URL/kwargs — not per event loop. Once one test's portal loop closes,
   any later test (or call) that resolves to the same cached filesystem
   instance fails with "Event loop is closed", because the cached instance
   is still holding a client bound to the now-dead loop.

Running genuinely async, pinned to the same session-scoped loop that
`zarr_server` itself uses, avoids both problems at the root: there is only
ever one event loop for the whole test session, so nothing tied to it is
ever torn down mid-suite.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import Datasource, DatasourceType
from cell_explorer_api.main import create_app

# See test_metadata_harvest.py / test_store_metadata.py for why this marker is
# required: zarr_server is a session-scoped async fixture, but the root
# pyproject.toml defaults async test loops to function scope.
pytestmark = pytest.mark.asyncio(loop_scope="session")

API_KEY = "test-admin-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture()
def app():
    return create_app(Settings(database_url="sqlite+aiosqlite://", admin_api_key=API_KEY))


@pytest_asyncio.fixture(loop_scope="session")
async def app_with_datasource(app, zarr_server):
    """App whose only datasource points at the fixture server."""
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    # expire_on_commit=False: without it, accessing ds.id after commit() below
    # triggers an implicit lazy-load refresh, which raises MissingGreenlet
    # under the async driver (see test_metadata_harvest.py's _engine_with_dataset
    # for the same pattern in this codebase).
    async with AsyncSession(engine, expire_on_commit=False) as session:
        ds = Datasource(
            name="Fixture server",
            type=DatasourceType.S3_CLOUDFRONT,
            base_url=zarr_server,
        )
        session.add(ds)
        await session.commit()
        datasource_id = str(ds.id)
    app.state.db_engine = engine
    return app, datasource_id


def _create_payload(datasource_id: str, path: str, slug: str = "tiny") -> dict:
    return {
        "datasource_id": datasource_id,
        "name": "Tiny",
        "slug": slug,
        "path": path,
        "is_public": True,
    }


async def test_create_harvests_metadata(app_with_datasource):
    app, datasource_id = app_with_datasource
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/admin/datasets",
            json=_create_payload(datasource_id, "tiny_v3.zarr"),
            headers=AUTH_HEADER,
        )
    assert res.status_code == 201
    metadata = res.json()["metadata"]
    assert metadata["status"] == "ok"
    assert metadata["n_obs"] == 12
    assert metadata["n_vars"] == 5


async def test_create_succeeds_when_store_unreachable(app_with_datasource):
    app, datasource_id = app_with_datasource
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/admin/datasets",
            json=_create_payload(datasource_id, "does_not_exist.zarr"),
            headers=AUTH_HEADER,
        )
    assert res.status_code == 201, "a failed harvest must not fail the write"
    metadata = res.json()["metadata"]
    assert metadata["status"] == "error"
    assert metadata["error"]
    assert metadata["n_obs"] is None


async def test_update_path_reharvests(app_with_datasource):
    app, datasource_id = app_with_datasource
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/api/admin/datasets",
            json=_create_payload(datasource_id, "does_not_exist.zarr"),
            headers=AUTH_HEADER,
        )
        res = await client.put(
            "/api/admin/datasets/tiny",
            json={"path": "tiny_v3.zarr"},
            headers=AUTH_HEADER,
        )
    assert res.status_code == 200
    assert res.json()["metadata"]["status"] == "ok"
    assert res.json()["metadata"]["n_obs"] == 12


async def test_update_unrelated_field_does_not_reharvest(app_with_datasource):
    app, datasource_id = app_with_datasource
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/api/admin/datasets",
            json=_create_payload(datasource_id, "tiny_v3.zarr"),
            headers=AUTH_HEADER,
        )
        before = (await client.get("/api/admin/datasets", headers=AUTH_HEADER)).json()
        attempted_at = before["datasets"][0]["metadata"]["last_attempt_at"]

        res = await client.put(
            "/api/admin/datasets/tiny", json={"name": "Renamed"}, headers=AUTH_HEADER
        )
    assert res.status_code == 200
    assert res.json()["metadata"]["last_attempt_at"] == attempted_at
