"""Tests for StrataStore."""

import numpy as np

from zarr_access import StrataStore
from zarr_access.strata_store import (
    AtomicStrataTable,
    CoarseStrataTable,
    StrataTable,
)


def test_coarse_strata_table_is_a_strata_table():
    """The discriminated union resolves to either coarse or atomic."""
    coarse: CoarseStrataTable = CoarseStrataTable(
        kind="coarse",
        slug="cell_type",
        axes=["cell_type"],
        stratum_keys=np.array([["A"], ["B"]]),
        gene_indices=None,
        sum_x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        sum_xx=np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32),
        nnz=np.array([[1, 1], [1, 1]], dtype=np.int32),
        n_cells=np.array([2, 2], dtype=np.int32),
        schema_version="1.0",
    )
    table: StrataTable = coarse
    assert table.kind == "coarse"


def test_atomic_strata_table_kind():
    """AtomicStrataTable has kind='atomic' and no slug field."""
    atomic = AtomicStrataTable(
        kind="atomic",
        axes=["cell_type", "donor"],
        stratum_keys=np.array([["A", "d1"]]),
        gene_indices=None,
        sum_x=np.array([[1.0]], dtype=np.float32),
        sum_xx=np.array([[1.0]], dtype=np.float32),
        nnz=np.array([[1]], dtype=np.int32),
        n_cells=np.array([1], dtype=np.int32),
        schema_version="1.0",
    )
    assert atomic.kind == "atomic"
    assert not hasattr(atomic, "slug")


def test_strata_store_class_exists():
    """StrataStore is exported from zarr_access."""
    assert StrataStore is not None
    assert callable(StrataStore)


import pytest_asyncio
from zarr_access import ZarrStore


@pytest_asyncio.fixture
async def strata_store(fixture_server):
    zarr_store = await ZarrStore.open(f"{fixture_server}/strata-tiny.zarr")
    return await StrataStore.open(zarr_store)


@pytest_asyncio.fixture
async def no_strata_store(fixture_server):
    """pbmc3k.zarr has no uns/strata/ group — discovery should return empties."""
    zarr_store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    return await StrataStore.open(zarr_store)


async def test_discovery_atomic(strata_store):
    assert strata_store.has_atomic() is True
    assert strata_store.atomic_axes() == ["cell_type", "donor"]
    assert strata_store.atomic_strata_count() == 6


async def test_discovery_coarse(strata_store):
    assert strata_store.coarse_slugs() == ["cell_type"]
    assert strata_store.coarse_axes("cell_type") == ["cell_type"]
    assert strata_store.coarse_strata_count("cell_type") == 3


async def test_coarse_axes_unknown_slug_raises(strata_store):
    import pytest
    with pytest.raises(KeyError, match="no_such_slug"):
        strata_store.coarse_axes("no_such_slug")


async def test_no_strata_returns_empty(no_strata_store):
    assert no_strata_store.has_atomic() is False
    assert no_strata_store.atomic_axes() is None
    assert no_strata_store.atomic_strata_count() is None
    assert no_strata_store.coarse_slugs() == []


async def test_read_coarse_round_trip(strata_store):
    coarse = await strata_store.read_coarse("cell_type")
    assert coarse.kind == "coarse"
    assert coarse.slug == "cell_type"
    assert coarse.axes == ["cell_type"]
    assert coarse.gene_indices is None
    assert coarse.schema_version == "1.0"

    # 3 strata × 10 genes
    assert coarse.stratum_keys.shape == (3, 1)
    assert coarse.sum_x.shape == (3, 10)
    assert coarse.sum_xx.shape == (3, 10)
    assert coarse.nnz.shape == (3, 10)
    assert coarse.n_cells.shape == (3,)

    # Cell counts: 20 (A) + 20 (B) + 10 (C) = 50
    assert int(coarse.n_cells.sum()) == 50

    # Stratum keys are 1-axis -> each row is a 1-element entry
    flat_keys = sorted(coarse.stratum_keys[:, 0].tolist())
    assert flat_keys == ["A", "B", "C"]


async def test_read_coarse_unknown_slug_raises(strata_store):
    import pytest
    with pytest.raises(KeyError, match="no_such_slug"):
        await strata_store.read_coarse("no_such_slug")


async def test_read_atomic_genes_slice(strata_store):
    atomic = await strata_store.read_atomic_genes([0, 5, 9])
    assert atomic.kind == "atomic"
    assert atomic.axes == ["cell_type", "donor"]
    assert atomic.gene_indices == [0, 5, 9]
    # 6 strata × 3 selected genes
    assert atomic.sum_x.shape == (6, 3)
    assert atomic.n_cells.shape == (6,)
    assert atomic.stratum_keys.shape == (6, 2)


async def test_read_atomic_genes_empty(strata_store):
    atomic = await strata_store.read_atomic_genes([])
    assert atomic.kind == "atomic"
    assert atomic.gene_indices == []
    # Gene-data arrays are empty
    assert atomic.sum_x.size == 0
    assert atomic.sum_xx.size == 0
    assert atomic.nnz.size == 0
    # But stratum_keys + n_cells are still populated
    assert atomic.stratum_keys.shape == (6, 2)
    assert atomic.n_cells.shape == (6,)


async def test_read_atomic_genes_no_atomic_raises(no_strata_store):
    import pytest
    with pytest.raises(ValueError, match="no atomic"):
        await no_strata_store.read_atomic_genes([0])
