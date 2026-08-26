"""The public dataset API's metadata field."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import (
    Dataset,
    DatasetMetadata,
    Datasource,
    DatasourceType,
)
from cell_explorer_api.main import create_app


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture()
def app():
    return create_app(Settings(database_url="sqlite+aiosqlite://"))


@pytest_asyncio.fixture()
async def seeded_app(app):
    """Three datasets: harvested, harvest-then-failed, and never harvested."""
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="CDN", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn.example.com"
        )
        session.add(ds)
        await session.flush()

        for slug in ("harvested", "stale", "never"):
            session.add(Dataset(
                datasource_id=ds.id, name=slug.title(), slug=slug,
                path=f"{slug}.zarr", is_public=True,
            ))
        await session.flush()

        ids = {d.slug: d.id for d in (await session.exec(select(Dataset))).all()}
        session.add(DatasetMetadata(
            dataset_id=ids["harvested"], n_obs=12, n_vars=5, zarr_version=3,
            obsm_keys=["X_umap"], obs_columns=["cell_type"], var_columns=["feature_name"],
            layers=["counts"], x_dtype="float32", x_encoding="array",
            fetched_at=_utcnow(), status="ok",
        ))
        session.add(DatasetMetadata(
            dataset_id=ids["stale"], n_obs=99, fetched_at=_utcnow(),
            status="error", error="boom at https://internal.example.com",
        ))
        session.add(DatasetMetadata(
            dataset_id=ids["never"], status="error", error="never worked",
        ))
        await session.commit()
    app.state.db_engine = engine
    return app


def _by_slug(client: TestClient) -> dict:
    res = client.get("/api/datasets")
    assert res.status_code == 200
    return {d["slug"]: d for d in res.json()["datasets"]}


def test_metadata_present_after_successful_harvest(seeded_app):
    datasets = _by_slug(TestClient(seeded_app))
    metadata = datasets["harvested"]["metadata"]
    assert metadata["n_obs"] == 12
    assert metadata["n_vars"] == 5
    assert metadata["obsm_keys"] == ["X_umap"]
    assert metadata["fetched_at"]


def test_metadata_retained_when_latest_attempt_failed(seeded_app):
    datasets = _by_slug(TestClient(seeded_app))
    assert datasets["stale"]["metadata"]["n_obs"] == 99


def test_metadata_null_when_never_harvested(seeded_app):
    datasets = _by_slug(TestClient(seeded_app))
    assert datasets["never"]["metadata"] is None


def test_status_and_error_never_exposed_publicly(seeded_app):
    body = TestClient(seeded_app).get("/api/datasets").text
    assert "internal.example.com" not in body
    for dataset in TestClient(seeded_app).get("/api/datasets").json()["datasets"]:
        metadata = dataset["metadata"]
        if metadata is not None:
            assert "status" not in metadata
            assert "error" not in metadata


def test_single_dataset_route_includes_metadata(seeded_app):
    res = TestClient(seeded_app).get("/api/datasets/harvested")
    assert res.status_code == 200
    assert res.json()["metadata"]["n_obs"] == 12
