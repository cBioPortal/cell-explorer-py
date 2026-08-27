"""Admin visibility into columns that matched no facet definition."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import Dataset, DatasetMetadata, Datasource, DatasourceType
from cell_explorer_api.main import create_app

API_KEY = "test-admin-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture()
def app():
    return create_app(Settings(database_url="sqlite+aiosqlite://", admin_api_key=API_KEY))


@pytest_asyncio.fixture()
async def seeded_app(app):
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        ds = Datasource(name="CDN", type=DatasourceType.S3_CLOUDFRONT,
                        base_url="https://cdn.example.com")
        session.add(ds)
        await session.flush()
        for slug in ("a", "b"):
            d = Dataset(datasource_id=ds.id, name=slug, slug=slug,
                        path=f"{slug}.zarr", is_public=True)
            session.add(d)
            await session.flush()
            session.add(DatasetMetadata(
                dataset_id=d.id, status="ok",
                obs_facets={
                    "tissue": {"dtype": "categorical", "cardinality": 3, "values": []},
                    "seurat_clusters": {"dtype": "categorical", "cardinality": 9, "values": []},
                },
            ))
        await session.commit()
    app.state.db_engine = engine
    return app


def test_lists_only_columns_matching_no_definition(seeded_app):
    res = TestClient(seeded_app).get("/api/admin/facets/unmapped", headers=AUTH_HEADER)
    assert res.status_code == 200
    names = {r["name"]: r["dataset_count"] for r in res.json()["columns"]}
    assert "seurat_clusters" in names
    assert "tissue" not in names, "a mapped column is not unmapped"


def test_counts_datasets_carrying_each_column(seeded_app):
    res = TestClient(seeded_app).get("/api/admin/facets/unmapped", headers=AUTH_HEADER)
    names = {r["name"]: r["dataset_count"] for r in res.json()["columns"]}
    assert names["seurat_clusters"] == 2


def test_requires_admin_auth(seeded_app):
    # require_admin returns 403 for a missing credential in this codebase.
    assert TestClient(seeded_app).get("/api/admin/facets/unmapped").status_code == 403
