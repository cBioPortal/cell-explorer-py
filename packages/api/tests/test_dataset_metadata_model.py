"""Tests for the DatasetMetadata table."""

from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import (
    Dataset,
    DatasetMetadata,
    Datasource,
    DatasourceType,
)


async def _engine():
    # create_engine attaches the SQLite foreign_keys pragma listener, which
    # ON DELETE CASCADE needs in tests.
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


async def _seed_dataset(session: AsyncSession) -> Dataset:
    ds = Datasource(
        name="Test CDN",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://cdn.example.com",
    )
    session.add(ds)
    await session.flush()
    dataset = Dataset(
        datasource_id=ds.id,
        name="Public Atlas",
        slug="public-atlas",
        path="datasets/public.zarr",
        is_public=True,
    )
    session.add(dataset)
    await session.flush()
    return dataset


async def test_dataset_metadata_round_trip():
    engine = await _engine()
    async with AsyncSession(engine) as session:
        dataset = await _seed_dataset(session)
        session.add(DatasetMetadata(
            dataset_id=dataset.id,
            n_obs=870000,
            n_vars=2000,
            zarr_version=3,
            obsm_keys=["X_umap"],
            obs_columns=["cell_type"],
            var_columns=["feature_name"],
            layers=["counts"],
            x_dtype="float32",
            x_encoding="array",
            status="ok",
        ))
        await session.commit()

        result = await session.exec(select(DatasetMetadata))
        row = result.one()
        assert row.n_obs == 870000
        assert row.obsm_keys == ["X_umap"]
        assert row.status == "ok"
        assert row.error is None
        assert row.last_attempt_at is not None


async def test_dataset_metadata_cascades_on_dataset_delete():
    engine = await _engine()
    async with AsyncSession(engine) as session:
        dataset = await _seed_dataset(session)
        session.add(DatasetMetadata(dataset_id=dataset.id, n_obs=10, status="ok"))
        await session.commit()

        await session.delete(dataset)
        await session.commit()

        result = await session.exec(select(DatasetMetadata))
        assert result.all() == []
