"""Tests for the dataset metadata harvester."""

import pytest
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import (
    Dataset,
    DatasetMetadata,
    Datasource,
    DatasourceType,
)
from cell_explorer_api.services.metadata_harvest import (
    harvest_dataset_metadata,
    store_harvest_result,
)

# See test_store_metadata.py for why this marker is required: zarr_server is a
# session-scoped async fixture, but the root pyproject.toml defaults async test
# loops to function scope. Without pinning these tests to loop_scope="session"
# the test run hangs forever waiting on the fixture's server.
pytestmark = pytest.mark.asyncio(loop_scope="session")


def _datasource(base_url: str) -> Datasource:
    return Datasource(
        name="Fixture server",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url=base_url,
    )


def _dataset(datasource: Datasource, path: str) -> Dataset:
    return Dataset(
        datasource_id=datasource.id,
        name="Tiny",
        slug="tiny",
        path=path,
        is_public=True,
    )


async def _engine_with_dataset():
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    # expire_on_commit=False: without it, accessing dataset.id after commit()
    # below triggers an implicit lazy-load refresh, which raises
    # MissingGreenlet under the async driver (see test_collections_model.py
    # and test_threads.py for the same pattern in this codebase).
    async with AsyncSession(engine, expire_on_commit=False) as session:
        ds = Datasource(
            name="CDN", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn.example.com"
        )
        session.add(ds)
        await session.flush()
        dataset = _dataset(ds, "tiny_v3.zarr")
        session.add(dataset)
        await session.commit()
        return engine, dataset.id


async def test_harvest_reads_public_store(zarr_server):
    datasource = _datasource(zarr_server)
    outcome = await harvest_dataset_metadata(_dataset(datasource, "tiny_v3.zarr"), datasource)
    assert outcome.status == "ok"
    assert outcome.error is None
    assert outcome.metadata.n_obs == 12


async def test_harvest_uses_internal_base_url_when_set(zarr_server):
    datasource = _datasource("https://unreachable.example.com")
    datasource.internal_base_url = zarr_server
    outcome = await harvest_dataset_metadata(_dataset(datasource, "tiny_v3.zarr"), datasource)
    assert outcome.status == "ok"


async def test_harvest_unreachable_store_returns_error(zarr_server):
    datasource = _datasource(zarr_server)
    outcome = await harvest_dataset_metadata(_dataset(datasource, "does_not_exist.zarr"), datasource)
    assert outcome.status == "error"
    assert outcome.error
    assert outcome.metadata is None


async def test_harvest_non_anndata_store_returns_error(zarr_server):
    datasource = _datasource(zarr_server)
    outcome = await harvest_dataset_metadata(_dataset(datasource, "not_anndata.zarr"), datasource)
    assert outcome.status == "error"
    assert outcome.metadata is None


async def test_harvest_private_without_credential_ref_returns_error(zarr_server):
    datasource = _datasource(zarr_server)  # credential_ref is None
    dataset = _dataset(datasource, "tiny_v3.zarr")
    dataset.is_public = False
    outcome = await harvest_dataset_metadata(dataset, datasource)
    assert outcome.status == "error"
    assert "credential_ref" in outcome.error


async def test_harvest_non_credential_error_during_mint_returns_error(zarr_server, monkeypatch):
    """mint_credentials can raise non-CredentialError exceptions too — e.g. an
    OSError reading a corrupt/unreadable HTTP_TOKEN private key file. Those must
    be caught and recorded, not propagated."""

    def _raise_unreadable_key(*_args, **_kwargs):
        raise OSError("key file unreadable")

    monkeypatch.setattr(
        "cell_explorer_api.services.metadata_harvest.mint_credentials",
        _raise_unreadable_key,
    )

    datasource = _datasource(zarr_server)
    dataset = _dataset(datasource, "tiny_v3.zarr")
    dataset.is_public = False
    outcome = await harvest_dataset_metadata(dataset, datasource)
    assert outcome.status == "error"
    assert outcome.metadata is None
    assert "key file unreadable" in outcome.error


async def test_store_result_writes_values_on_success(zarr_server):
    engine, dataset_id = await _engine_with_dataset()
    datasource = _datasource(zarr_server)
    outcome = await harvest_dataset_metadata(_dataset(datasource, "tiny_v3.zarr"), datasource)

    async with AsyncSession(engine) as session:
        await store_harvest_result(session, dataset_id, outcome)
        await session.commit()
        row = (await session.exec(select(DatasetMetadata))).one()

    assert row.status == "ok"
    assert row.n_obs == 12
    assert row.fetched_at is not None
    assert row.error is None


async def test_failed_harvest_preserves_previous_values(zarr_server):
    engine, dataset_id = await _engine_with_dataset()
    datasource = _datasource(zarr_server)

    good = await harvest_dataset_metadata(_dataset(datasource, "tiny_v3.zarr"), datasource)
    bad = await harvest_dataset_metadata(_dataset(datasource, "gone.zarr"), datasource)

    async with AsyncSession(engine) as session:
        await store_harvest_result(session, dataset_id, good)
        await session.commit()
        first = (await session.exec(select(DatasetMetadata))).one()
        first_fetched_at = first.fetched_at

        await store_harvest_result(session, dataset_id, bad)
        await session.commit()
        session.expire_all()
        row = (await session.exec(select(DatasetMetadata))).one()

    assert row.status == "error"
    assert row.error
    assert row.n_obs == 12, "a failed harvest must not blank previously good values"
    assert row.fetched_at == first_fetched_at, "fetched_at tracks the last SUCCESS"


async def test_stores_obs_facets_on_success(zarr_server):
    engine, dataset_id = await _engine_with_dataset()
    datasource = _datasource(zarr_server)
    outcome = await harvest_dataset_metadata(_dataset(datasource, "tiny_v3.zarr"), datasource)

    async with AsyncSession(engine) as session:
        await store_harvest_result(session, dataset_id, outcome)
        await session.commit()
        row = (await session.exec(select(DatasetMetadata))).one()

    assert sorted(row.obs_facets["tissue"]["values"]) == ["brain", "liver", "lung"]
    assert row.obs_facets["tissue"]["cardinality"] == 3
    assert row.obs_facets["n_counts"]["values"] is None


async def test_failed_harvest_preserves_obs_facets(zarr_server):
    # Same rule as the other value columns: a transient outage must not blank
    # facets the catalogue is already serving.
    engine, dataset_id = await _engine_with_dataset()
    datasource = _datasource(zarr_server)
    good = await harvest_dataset_metadata(_dataset(datasource, "tiny_v3.zarr"), datasource)
    bad = await harvest_dataset_metadata(_dataset(datasource, "gone.zarr"), datasource)

    async with AsyncSession(engine) as session:
        await store_harvest_result(session, dataset_id, good)
        await session.commit()
        await store_harvest_result(session, dataset_id, bad)
        await session.commit()
        session.expire_all()
        row = (await session.exec(select(DatasetMetadata))).one()

    assert row.status == "error"
    assert row.obs_facets["tissue"]["cardinality"] == 3


async def test_clear_stale_values_clears_obs_facets(zarr_server):
    # When the dataset's path changed, retained facets describe a store the
    # dataset no longer points at — they must go with the counts.
    engine, dataset_id = await _engine_with_dataset()
    datasource = _datasource(zarr_server)
    good = await harvest_dataset_metadata(_dataset(datasource, "tiny_v3.zarr"), datasource)
    bad = await harvest_dataset_metadata(_dataset(datasource, "gone.zarr"), datasource)

    async with AsyncSession(engine) as session:
        await store_harvest_result(session, dataset_id, good)
        await session.commit()
        await store_harvest_result(session, dataset_id, bad, clear_stale_values=True)
        await session.commit()
        session.expire_all()
        row = (await session.exec(select(DatasetMetadata))).one()

    assert row.obs_facets == {}
    assert row.n_obs is None
