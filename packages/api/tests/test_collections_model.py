"""Collection model and its relationship to Dataset."""

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import Collection, Dataset, Datasource, DatasourceType


@pytest_asyncio.fixture()
async def session():
    """Async SQLite session with foreign_keys=ON.

    Uses the project's create_engine helper so the SQLite-FK pragma listener
    from db/__init__.py is attached — otherwise ON DELETE SET NULL would
    silently no-op. expire_on_commit=False avoids MissingGreenlet errors when
    a test accesses attributes of an object loaded before a later commit
    (see tests/services/test_threads.py for the same pattern).
    """
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s


async def _datasource(session) -> Datasource:
    ds = Datasource(
        name="Test CDN",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://cdn.example.com",
    )
    session.add(ds)
    await session.commit()
    await session.refresh(ds)
    return ds


@pytest.mark.asyncio
async def test_dataset_collection_id_defaults_to_none(session):
    ds = await _datasource(session)
    dataset = Dataset(datasource_id=ds.id, name="Solo", slug="solo", path="a.zarr")
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    assert dataset.collection_id is None


@pytest.mark.asyncio
async def test_dataset_can_join_a_collection(session):
    ds = await _datasource(session)
    collection = Collection(name="A Study", slug="a-study", description="About it")
    session.add(collection)
    await session.commit()
    await session.refresh(collection)

    dataset = Dataset(
        datasource_id=ds.id,
        name="Sample 1",
        slug="a-study-sample-1",
        path="a.zarr",
        collection_id=collection.id,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)

    assert dataset.collection_id == collection.id
    assert collection.publication_url is None
    assert collection.publication_citation is None


@pytest.mark.asyncio
async def test_collection_slug_is_unique(session):
    session.add(Collection(name="One", slug="dupe"))
    await session.commit()
    session.add(Collection(name="Two", slug="dupe"))
    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.asyncio
async def test_deleting_collection_nulls_dataset_collection_id(session):
    ds = await _datasource(session)
    collection = Collection(name="Doomed", slug="doomed")
    session.add(collection)
    await session.commit()
    await session.refresh(collection)

    dataset = Dataset(
        datasource_id=ds.id,
        name="Survivor",
        slug="survivor",
        path="a.zarr",
        collection_id=collection.id,
    )
    session.add(dataset)
    await session.commit()

    await session.delete(collection)
    await session.commit()

    result = await session.exec(select(Dataset).where(Dataset.slug == "survivor"))
    survivor = result.first()
    assert survivor is not None, "deleting a collection must not delete its datasets"
    assert survivor.collection_id is None
