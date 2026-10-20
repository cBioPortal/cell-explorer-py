"""The catalog and chat tables on Postgres, migrated by Alembic.

The rest of the suite runs on in-memory SQLite, which accepts values Postgres
rejects. Runs only when TEST_POSTGRES_URL points at an empty, disposable
database, e.g. postgresql+asyncpg://cell_explorer:test@localhost:5432/cell_explorer.
Every run drops and recreates its public schema.
"""

import asyncio
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db import create_engine
from cell_explorer_api.db.copy_to_postgres import CopyError, copy
from cell_explorer_api.db.models import (
    ChatFeedback,
    ChatMessageRow,
    ChatThread,
    Collection,
    Dataset,
    DatasetMetadata,
    Datasource,
    DatasourceType,
)
from cell_explorer_api.main import create_app

POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="TEST_POSTGRES_URL not set")

API_ROOT = Path(__file__).resolve().parents[1]
API_KEY = "test-admin-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


async def _reset_schema(url: str) -> None:
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


def _upgrade(url: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", "head"],
        cwd=API_ROOT,
        env={**os.environ, "DATABASE_URL": url},
        check=True,
    )


@pytest.fixture()
def migrated_url() -> str:
    """An empty database migrated to head the way the image does on boot."""
    asyncio.run(_reset_schema(POSTGRES_URL))
    _upgrade(POSTGRES_URL)
    return POSTGRES_URL


@pytest.fixture()
def sqlite_url(tmp_path) -> str:
    """An empty SQLite catalog migrated to head, like a deployment's data volume."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'cell_explorer.db'}"
    _upgrade(url)
    return url


def test_admin_catalog_round_trip(migrated_url):
    app = create_app(Settings(database_url=migrated_url, admin_api_key=API_KEY))
    with TestClient(app) as client:
        r = client.post(
            "/api/admin/datasources",
            json={"name": "CDN", "type": "s3_cloudfront", "base_url": "https://cdn.example.com"},
            headers=AUTH_HEADER,
        )
        assert r.status_code == 201, r.text
        datasource_id = r.json()["id"]
        r = client.put(
            f"/api/admin/datasources/{datasource_id}",
            json={"name": "CDN renamed"},
            headers=AUTH_HEADER,
        )
        assert r.status_code == 200, r.text

        r = client.post(
            "/api/admin/collections",
            json={"name": "Atlas", "slug": "atlas"},
            headers=AUTH_HEADER,
        )
        assert r.status_code == 201, r.text
        collection_id = r.json()["id"]
        r = client.put(
            "/api/admin/collections/atlas",
            json={"description": "Updated"},
            headers=AUTH_HEADER,
        )
        assert r.status_code == 200, r.text

        r = client.post(
            "/api/admin/datasets",
            json={
                "datasource_id": datasource_id,
                "collection_id": collection_id,
                "name": "Public Atlas",
                "slug": "public-atlas",
                "path": "datasets/public.zarr",
                "is_public": True,
                "required_roles": ["lab-a"],
                "default_view": {"pointSize": 3},
            },
            headers=AUTH_HEADER,
        )
        assert r.status_code == 201, r.text
        r = client.put(
            "/api/admin/datasets/public-atlas",
            json={"name": "Public Atlas v2", "chat_enabled": True},
            headers=AUTH_HEADER,
        )
        assert r.status_code == 200, r.text

        r = client.post(
            "/api/admin/datasets",
            json={
                "datasource_id": datasource_id,
                "name": "Duplicate",
                "slug": "public-atlas",
                "path": "datasets/dup.zarr",
            },
            headers=AUTH_HEADER,
        )
        assert r.status_code == 409, "unique slug enforced"

        datasets = {d["slug"]: d for d in client.get("/api/datasets").json()["datasets"]}
        assert datasets["public-atlas"]["name"] == "Public Atlas v2"
        assert [c["slug"] for c in client.get("/api/collections").json()["collections"]] == ["atlas"]

        # Deleting the collection orphans its dataset rather than deleting it.
        assert client.delete("/api/admin/collections/atlas", headers=AUTH_HEADER).status_code == 204
        assert client.get("/api/datasets/public-atlas").status_code == 200
        assert client.delete("/api/admin/datasets/public-atlas", headers=AUTH_HEADER).status_code == 204
        assert client.get("/api/datasets/public-atlas").status_code == 404


async def test_chat_and_metadata_rows(migrated_url):
    engine = create_engine(migrated_url)
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            ds = Datasource(name="CDN", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn")
            session.add(ds)
            await session.flush()
            dataset = Dataset(datasource_id=ds.id, name="A", slug="a", path="a.zarr")
            session.add(dataset)
            await session.flush()
            session.add(DatasetMetadata(
                dataset_id=dataset.id, status="ok", n_obs=10,
                obs_facets={"cell_type": {"dtype": "categorical", "cardinality": 3}},
            ))
            thread = ChatThread(user_sub="user-1", dataset_id=dataset.id, title="t")
            session.add(thread)
            await session.flush()
            message = ChatMessageRow(thread_id=thread.id, role="assistant", content="hi")
            session.add(message)
            await session.flush()
            session.add(ChatFeedback(message_id=message.id, user_sub="user-1", rating="up"))
            await session.commit()

        async with AsyncSession(engine) as session:
            meta = (await session.exec(select(DatasetMetadata))).one()
            assert meta.obs_facets["cell_type"]["cardinality"] == 3

            session.add(ChatFeedback(message_id=message.id, user_sub="user-1", rating="down"))
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()

            # Deleting a dataset cascades to its metadata and its whole chat history.
            await session.delete(await session.get(Dataset, dataset.id))
            await session.commit()
            for model in (DatasetMetadata, ChatThread, ChatMessageRow, ChatFeedback):
                assert (await session.exec(select(model))).all() == [], model.__name__
    finally:
        await engine.dispose()


async def test_copy_from_sqlite(sqlite_url, migrated_url):
    source = create_engine(sqlite_url)
    try:
        async with AsyncSession(source, expire_on_commit=False) as session:
            ds = Datasource(name="CDN", type=DatasourceType.HTTP_TOKEN, base_url="https://cdn")
            collection = Collection(name="Atlas", slug="atlas")
            session.add_all([ds, collection])
            await session.flush()
            dataset = Dataset(
                datasource_id=ds.id, collection_id=collection.id, name="A", slug="a",
                path="a.zarr", required_roles=["lab-a"], default_view={"pointSize": 3},
                prompt_addendum="x" * 5000,
            )
            session.add(dataset)
            await session.flush()
            session.add(DatasetMetadata(
                dataset_id=dataset.id, status="ok", n_obs=10, obs_columns=["cell_type"],
                obs_facets={"cell_type": {"dtype": "categorical", "cardinality": 3}},
            ))
            thread = ChatThread(user_sub="user-1", dataset_id=dataset.id, title="t")
            session.add(thread)
            await session.flush()
            message = ChatMessageRow(thread_id=thread.id, role="assistant", content="hi ✓")
            session.add(message)
            await session.flush()
            session.add(ChatFeedback(message_id=message.id, user_sub="user-1", rating="up"))
            await session.commit()
    finally:
        await source.dispose()

    counts = await copy(sqlite_url, migrated_url)
    assert counts == {
        "datasources": 1, "collections": 1, "datasets": 1, "dataset_metadata": 1,
        "chat_threads": 1, "chat_messages": 1, "chat_feedback": 1,
    }

    target = create_engine(migrated_url)
    try:
        async with AsyncSession(target) as session:
            copied = await session.get(Dataset, dataset.id)
            assert copied.collection_id == collection.id
            assert copied.default_view == {"pointSize": 3}
            assert copied.created_at == dataset.created_at
            assert (await session.get(Datasource, ds.id)).type is DatasourceType.HTTP_TOKEN
            assert (await session.get(ChatMessageRow, message.id)).content == "hi ✓"
    finally:
        await target.dispose()

    with pytest.raises(CopyError, match="not empty"):
        await copy(sqlite_url, migrated_url)


async def test_copy_refuses_orphaned_rows(sqlite_url, migrated_url):
    # A connection without the foreign-key pragma, as old SQLite writes were.
    source = create_async_engine(sqlite_url)
    try:
        async with AsyncSession(source) as session:
            session.add(ChatThread(user_sub="user-1", dataset_id=uuid.uuid4(), title="t"))
            await session.commit()
    finally:
        await source.dispose()

    with pytest.raises(CopyError, match="chat_threads.dataset_id: 1 row"):
        await copy(sqlite_url, migrated_url)
