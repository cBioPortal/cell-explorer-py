"""List endpoints return rows oldest first, on both SQLite and Postgres.

Without an ORDER BY, SQLite happens to return insertion order and Postgres
returns no defined order at all, so the catalogue's default order would change
when a deployment moves to Postgres. Rows here are inserted out of creation
order, so a list that falls back to storage order fails.

The Postgres case runs only when TEST_POSTGRES_URL points at an empty,
disposable database (see test_postgres.py); it drops the public schema.
"""

import asyncio
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import Collection, Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app

POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")
API_ROOT = Path(__file__).resolve().parents[1]
API_KEY = "test-admin-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


def _at(day: int) -> datetime:
    return datetime(2026, 1, day, 12, 0, 0)


async def _reset_schema(url: str) -> None:
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


@pytest.fixture(params=["sqlite", "postgres"])
def db_url(request, tmp_path) -> str:
    """An empty database of each kind, migrated to head."""
    if request.param == "sqlite":
        url = f"sqlite+aiosqlite:///{tmp_path / 'cell_explorer.db'}"
    else:
        if not POSTGRES_URL:
            pytest.skip("TEST_POSTGRES_URL not set")
        url = POSTGRES_URL
        asyncio.run(_reset_schema(url))
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", "head"],
        cwd=API_ROOT,
        env={**os.environ, "DATABASE_URL": url},
        check=True,
    )
    return url


async def _seed(url: str) -> None:
    """Insert rows out of creation order, so storage order differs from it."""
    engine = create_engine(url)
    try:
        async with AsyncSession(engine) as session:
            newer_ds = Datasource(name="newer", type=DatasourceType.S3_CLOUDFRONT,
                                  base_url="https://b", created_at=_at(2))
            older_ds = Datasource(name="older", type=DatasourceType.S3_CLOUDFRONT,
                                  base_url="https://a", created_at=_at(1))
            newer_c = Collection(name="Newer", slug="newer", created_at=_at(2))
            older_c = Collection(name="Older", slug="older", created_at=_at(1))
            session.add_all([newer_ds, older_ds, newer_c, older_c])
            await session.flush()
            for day in (3, 1, 2):
                session.add(Dataset(
                    datasource_id=older_ds.id, collection_id=older_c.id, name=f"d{day}",
                    slug=f"d{day}", path=f"d{day}.zarr", is_public=True, created_at=_at(day),
                ))
            await session.commit()
    finally:
        await engine.dispose()


def test_lists_are_oldest_first(db_url):
    asyncio.run(_seed(db_url))
    app = create_app(Settings(database_url=db_url, admin_api_key=API_KEY))
    with TestClient(app) as client:
        def slugs(path: str, key: str, field: str = "slug") -> list[str]:
            r = client.get(path, headers=AUTH_HEADER)
            assert r.status_code == 200, r.text
            return [row[field] for row in r.json()[key]]

        assert slugs("/api/datasets", "datasets") == ["d1", "d2", "d3"]
        assert slugs("/api/collections/older", "datasets") == ["d1", "d2", "d3"]
        assert slugs("/api/admin/datasets", "datasets") == ["d1", "d2", "d3"]
        assert slugs("/api/admin/collections", "collections") == ["older", "newer"]
        assert slugs("/api/admin/datasources", "datasources", "name") == ["older", "newer"]
