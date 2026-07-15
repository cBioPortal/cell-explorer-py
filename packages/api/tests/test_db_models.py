"""Tests for database models."""

import uuid
from datetime import datetime, timezone

import pytest
from sqlmodel import SQLModel

from cell_explorer_api.db.models import ChatFeedback, Datasource, DatasourceType, Dataset


def test_datasource_create():
    ds = Datasource(
        name="MSK CloudFront",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://d1234.cloudfront.net",
        credential_ref="MSK_CLOUDFRONT",
    )
    assert ds.name == "MSK CloudFront"
    assert ds.type == DatasourceType.S3_CLOUDFRONT
    assert ds.base_url == "https://d1234.cloudfront.net"
    assert ds.credential_ref == "MSK_CLOUDFRONT"
    assert ds.id is not None
    assert isinstance(ds.created_at, datetime)


def test_datasource_credential_ref_nullable():
    ds = Datasource(
        name="Public CDN",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://public.cdn.net",
    )
    assert ds.credential_ref is None


def test_datasource_fetch_base_url_falls_back_to_base_url():
    ds = Datasource(
        name="No split",
        type=DatasourceType.HTTP_TOKEN,
        base_url="https://public.example",
    )
    assert ds.internal_base_url is None
    assert ds.fetch_base_url == "https://public.example"


def test_datasource_fetch_base_url_prefers_internal():
    ds = Datasource(
        name="Compose split",
        type=DatasourceType.HTTP_TOKEN,
        base_url="http://localhost:8002",
        internal_base_url="http://zarr-server:8000",
    )
    assert ds.fetch_base_url == "http://zarr-server:8000"


def test_dataset_create():
    datasource_id = uuid.uuid4()
    dataset = Dataset(
        datasource_id=datasource_id,
        name="BRCA Tumor Atlas",
        slug="brca-tumor-atlas",
        path="datasets/brca.zarr",
        is_public=True,
    )
    assert dataset.name == "BRCA Tumor Atlas"
    assert dataset.slug == "brca-tumor-atlas"
    assert dataset.path == "datasets/brca.zarr"
    assert dataset.is_public is True
    assert dataset.required_roles == []
    assert dataset.description is None


def test_dataset_with_roles():
    datasource_id = uuid.uuid4()
    dataset = Dataset(
        datasource_id=datasource_id,
        name="Private Dataset",
        slug="private-dataset",
        path="datasets/private.zarr",
        is_public=False,
        required_roles=["lab-smith", "project-alpha"],
    )
    assert dataset.required_roles == ["lab-smith", "project-alpha"]


def test_dataset_url_property():
    ds = Datasource(
        name="CDN",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://d1234.cloudfront.net",
    )
    dataset = Dataset(
        datasource_id=ds.id,
        name="Test",
        slug="test",
        path="datasets/test.zarr",
        is_public=True,
    )
    dataset.datasource = ds
    assert dataset.url == "https://d1234.cloudfront.net/datasets/test.zarr"


def test_datasource_type_enum():
    assert DatasourceType.S3_CLOUDFRONT == "s3_cloudfront"
    assert DatasourceType.HTTP_TOKEN == "http_token"


def test_chat_feedback_create():
    f = ChatFeedback(
        message_id=uuid.uuid4(),
        user_sub="user-1",
        rating="up",
    )
    assert f.rating == "up"
    assert f.comment is None
    assert f.id is not None
    assert isinstance(f.created_at, datetime)
    assert isinstance(f.updated_at, datetime)


def test_chat_feedback_accepts_comment():
    f = ChatFeedback(
        message_id=uuid.uuid4(),
        user_sub="user-1",
        rating="down",
        comment="off-topic",
    )
    assert f.rating == "down"
    assert f.comment == "off-topic"


@pytest.mark.asyncio
async def test_dataset_default_view_persists():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import SQLModel, select
    from sqlmodel.ext.asyncio.session import AsyncSession

    from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        ds = Datasource(name="cdn", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn")
        session.add(ds)
        await session.commit()
        await session.refresh(ds)

        dataset = Dataset(
            datasource_id=ds.id,
            name="D",
            slug="d",
            path="d.zarr",
            default_view={"colorBy": "category", "category": "cell_type"},
        )
        session.add(dataset)
        await session.commit()

        result = await session.exec(select(Dataset).where(Dataset.slug == "d"))
        loaded = result.first()
        assert loaded.default_view == {"colorBy": "category", "category": "cell_type"}


@pytest.mark.asyncio
async def test_dataset_default_view_defaults_to_none():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import SQLModel
    from sqlmodel.ext.asyncio.session import AsyncSession

    from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        ds = Datasource(name="cdn", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn")
        session.add(ds)
        await session.commit()
        await session.refresh(ds)
        dataset = Dataset(datasource_id=ds.id, name="D", slug="d2", path="d.zarr")
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)
        assert dataset.default_view is None
