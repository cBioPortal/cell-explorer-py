"""Public collection endpoints and their derived visibility."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import (
    Collection,
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
    """One public collection, one entirely private collection."""
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="CDN", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn.example.com"
        )
        open_c = Collection(
            name="Open Study",
            slug="open-study",
            description="Anyone may read this",
            publication_url="https://example.org/paper",
            publication_citation="Author et al. 2025",
        )
        shut_c = Collection(name="Gated Study", slug="gated-study")
        session.add_all([ds, open_c, shut_c])
        await session.commit()
        for obj in (ds, open_c, shut_c):
            await session.refresh(obj)

        session.add_all([
            Dataset(datasource_id=ds.id, collection_id=open_c.id, name="Public A",
                    slug="public-a", path="a.zarr", is_public=True),
            Dataset(datasource_id=ds.id, collection_id=open_c.id, name="Private B",
                    slug="private-b", path="b.zarr", is_public=False,
                    required_roles=["secret-role"]),
            Dataset(datasource_id=ds.id, collection_id=shut_c.id, name="Private C",
                    slug="private-c", path="c.zarr", is_public=False,
                    required_roles=["secret-role"]),
            Dataset(datasource_id=ds.id, name="Ungrouped", slug="ungrouped",
                    path="d.zarr", is_public=True),
        ])
        await session.commit()

        dataset_ids = {d.slug: d.id for d in (await session.exec(select(Dataset))).all()}
        session.add(DatasetMetadata(
            dataset_id=dataset_ids["public-a"], n_obs=12, n_vars=5, zarr_version=3,
            obsm_keys=["X_umap"], obs_columns=["cell_type"], var_columns=["feature_name"],
            layers=["counts"], x_dtype="float32", x_encoding="array",
            fetched_at=_utcnow(), status="ok",
        ))
        await session.commit()
    app.state.db_engine = engine
    return app


def test_anonymous_sees_only_collections_with_accessible_datasets(seeded_app):
    client = TestClient(seeded_app)
    body = client.get("/api/collections").json()
    slugs = [c["slug"] for c in body["collections"]]
    assert slugs == ["open-study"], "a collection with no accessible datasets must be hidden"


def test_dataset_count_reflects_only_accessible_datasets(seeded_app):
    client = TestClient(seeded_app)
    body = client.get("/api/collections").json()
    assert body["collections"][0]["dataset_count"] == 1


def test_collection_detail_returns_metadata_and_accessible_datasets(seeded_app):
    client = TestClient(seeded_app)
    body = client.get("/api/collections/open-study").json()
    assert body["name"] == "Open Study"
    assert body["publication_url"] == "https://example.org/paper"
    assert body["publication_citation"] == "Author et al. 2025"
    assert [d["slug"] for d in body["datasets"]] == ["public-a"]


def test_collection_detail_carries_dataset_metadata(seeded_app):
    client = TestClient(seeded_app)
    body = client.get("/api/collections/open-study").json()
    dataset = next(d for d in body["datasets"] if d["slug"] == "public-a")
    assert dataset["metadata"] is not None, (
        "the collection page renders the same table as the catalogue and needs the same counts"
    )
    assert dataset["metadata"]["n_obs"] == 12, (
        "the collection page renders the same table as the catalogue and needs the same counts"
    )


def test_fully_gated_collection_is_404_not_403(seeded_app):
    client = TestClient(seeded_app)
    response = client.get("/api/collections/gated-study")
    assert response.status_code == 404, "403 would confirm the collection exists"


def test_unknown_slug_is_404(seeded_app):
    client = TestClient(seeded_app)
    assert client.get("/api/collections/no-such-thing").status_code == 404


def test_authenticated_with_role_sees_previously_gated_collection(seeded_app):
    """The other half of derived visibility: revealed once the role matches."""
    seeded_app.dependency_overrides[optional_auth] = lambda: User(
        sub="user-1", roles=["secret-role"]
    )
    try:
        client = TestClient(seeded_app)
        body = client.get("/api/collections").json()
        slugs = [c["slug"] for c in body["collections"]]
        assert "gated-study" in slugs

        response = client.get("/api/collections/gated-study")
        assert response.status_code == 200
        assert [d["slug"] for d in response.json()["datasets"]] == ["private-c"]
    finally:
        seeded_app.dependency_overrides.pop(optional_auth, None)


def test_authenticated_with_role_expands_dataset_count_and_datasets(seeded_app):
    """A mixed collection gains its role-gated dataset once the caller qualifies."""
    seeded_app.dependency_overrides[optional_auth] = lambda: User(
        sub="user-1", roles=["secret-role"]
    )
    try:
        client = TestClient(seeded_app)
        body = client.get("/api/collections").json()
        open_summary = next(c for c in body["collections"] if c["slug"] == "open-study")
        assert open_summary["dataset_count"] == 2

        detail = client.get("/api/collections/open-study").json()
        assert sorted(d["slug"] for d in detail["datasets"]) == ["private-b", "public-a"]
    finally:
        seeded_app.dependency_overrides.pop(optional_auth, None)
