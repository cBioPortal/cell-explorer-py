"""Admin CRUD for collections."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Collection, Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app

API_KEY = "test-admin-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture()
def app():
    return create_app(Settings(database_url="sqlite+aiosqlite://", admin_api_key=API_KEY))


@pytest_asyncio.fixture()
async def ready_app(app):
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    app.state.db_engine = engine
    return app


def test_create_collection(ready_app):
    client = TestClient(ready_app)
    response = client.post(
        "/api/admin/collections",
        json={
            "name": "A Study",
            "slug": "a-study",
            "description": "About it",
            "publication_url": "https://example.org/paper",
            "publication_citation": "Author et al. 2025",
        },
        headers=AUTH_HEADER,
    )
    assert response.status_code == 201
    body = response.json()
    assert body["slug"] == "a-study"
    assert body["publication_citation"] == "Author et al. 2025"


def test_create_collection_requires_admin(ready_app):
    client = TestClient(ready_app)
    response = client.post(
        "/api/admin/collections", json={"name": "X", "slug": "x"}
    )
    assert response.status_code in (401, 403)


def test_list_collections(ready_app):
    client = TestClient(ready_app)
    client.post("/api/admin/collections", json={"name": "One", "slug": "one"},
                headers=AUTH_HEADER)
    body = client.get("/api/admin/collections", headers=AUTH_HEADER).json()
    assert [c["slug"] for c in body["collections"]] == ["one"]


def test_update_collection(ready_app):
    client = TestClient(ready_app)
    client.post("/api/admin/collections", json={"name": "Old", "slug": "c"},
                headers=AUTH_HEADER)
    response = client.put(
        "/api/admin/collections/c",
        json={"name": "New", "description": "Now described"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    assert response.json()["name"] == "New"
    assert response.json()["description"] == "Now described"


def test_update_unknown_collection_is_404(ready_app):
    client = TestClient(ready_app)
    response = client.put(
        "/api/admin/collections/nope", json={"name": "X"}, headers=AUTH_HEADER
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_collection_orphans_but_keeps_datasets(ready_app):
    client = TestClient(ready_app)
    created = client.post(
        "/api/admin/collections", json={"name": "Doomed", "slug": "doomed"},
        headers=AUTH_HEADER,
    ).json()

    engine = ready_app.state.db_engine
    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="CDN", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn.example.com"
        )
        session.add(ds)
        await session.commit()
        await session.refresh(ds)
        datasource_id = str(ds.id)

    client.post(
        "/api/admin/datasets",
        json={
            "datasource_id": datasource_id,
            "collection_id": created["id"],
            "name": "Survivor",
            "slug": "survivor",
            "path": "a.zarr",
        },
        headers=AUTH_HEADER,
    )

    assert client.delete("/api/admin/collections/doomed", headers=AUTH_HEADER).status_code == 204

    async with AsyncSession(engine) as session:
        result = await session.exec(select(Dataset).where(Dataset.slug == "survivor"))
        survivor = result.first()
    assert survivor is not None, "deleting a collection must not delete its datasets"
    assert survivor.collection_id is None
