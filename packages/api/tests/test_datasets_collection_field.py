"""GET /api/datasets exposes each dataset's collection."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Collection, Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app


@pytest.fixture()
def app():
    return create_app(Settings(database_url="sqlite+aiosqlite://"))


@pytest_asyncio.fixture()
async def seeded_app(app):
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="CDN", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn.example.com"
        )
        collection = Collection(name="A Study", slug="a-study")
        session.add(ds)
        session.add(collection)
        await session.commit()
        await session.refresh(ds)
        await session.refresh(collection)
        session.add(
            Dataset(
                datasource_id=ds.id,
                collection_id=collection.id,
                name="In Study",
                slug="in-study",
                path="a.zarr",
                is_public=True,
            )
        )
        session.add(
            Dataset(
                datasource_id=ds.id,
                name="Ungrouped",
                slug="ungrouped",
                path="b.zarr",
                is_public=True,
            )
        )
        await session.commit()
    app.state.db_engine = engine
    return app


def test_dataset_in_a_collection_reports_it(seeded_app):
    client = TestClient(seeded_app)
    datasets = {d["slug"]: d for d in client.get("/api/datasets").json()["datasets"]}
    assert datasets["in-study"]["collection"] == {"slug": "a-study", "name": "A Study"}


def test_dataset_without_a_collection_reports_none(seeded_app):
    client = TestClient(seeded_app)
    datasets = {d["slug"]: d for d in client.get("/api/datasets").json()["datasets"]}
    assert datasets["ungrouped"]["collection"] is None


def test_get_dataset_in_a_collection_reports_it(seeded_app):
    client = TestClient(seeded_app)
    dataset = client.get("/api/datasets/in-study").json()
    assert dataset["collection"] == {"slug": "a-study", "name": "A Study"}


def test_get_dataset_without_a_collection_reports_none(seeded_app):
    client = TestClient(seeded_app)
    dataset = client.get("/api/datasets/ungrouped").json()
    assert dataset["collection"] is None
