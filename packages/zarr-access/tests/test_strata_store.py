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


async def test_read_atomic_full(strata_store):
    atomic = await strata_store.read_atomic()
    assert atomic.kind == "atomic"
    assert atomic.gene_indices is None
    assert atomic.axes == ["cell_type", "donor"]
    # 6 strata × 10 genes
    assert atomic.sum_x.shape == (6, 10)
    assert atomic.n_cells.shape == (6,)
    assert int(atomic.n_cells.sum()) == 50


async def test_read_atomic_no_atomic_raises(no_strata_store):
    import pytest
    with pytest.raises(ValueError, match="no atomic"):
        await no_strata_store.read_atomic()


async def test_read_atomic_stratum_keys_returns_keys_and_n_cells(strata_store):
    keys, n_cells = await strata_store.read_atomic_stratum_keys()
    # Fixture: 6 strata x 2 axes
    assert keys.shape == (6, 2)
    assert n_cells.shape == (6,)
    # 50 total cells in the strata-tiny fixture
    assert int(n_cells.sum()) == 50


async def test_read_atomic_stratum_keys_no_atomic_raises(no_strata_store):
    import pytest
    with pytest.raises(ValueError, match="no atomic"):
        await no_strata_store.read_atomic_stratum_keys()


async def test_read_atomic_rows_slice(strata_store):
    # Pick a subset of the 6 fixture strata rows.
    atomic = await strata_store.read_atomic_rows([0, 2, 5])
    assert atomic.kind == "atomic"
    assert atomic.axes == ["cell_type", "donor"]
    # 3 selected rows, all 10 genes
    assert atomic.sum_x.shape == (3, 10)
    assert atomic.sum_xx.shape == (3, 10)
    assert atomic.nnz.shape == (3, 10)
    # stratum_keys and n_cells are sliced to the requested rows
    assert atomic.stratum_keys.shape == (3, 2)
    assert atomic.n_cells.shape == (3,)


async def test_read_atomic_rows_matches_full_table(strata_store):
    """Row-selective read must produce byte-identical values to slicing
    the full read. This is the regression guard against drift between
    the fancy-indexed read path and the bulk read path.
    """
    full = await strata_store.read_atomic()
    rows = [1, 3, 4]
    slim = await strata_store.read_atomic_rows(rows)

    np.testing.assert_array_equal(slim.sum_x, full.sum_x[rows])
    np.testing.assert_array_equal(slim.sum_xx, full.sum_xx[rows])
    np.testing.assert_array_equal(slim.nnz, full.nnz[rows])
    np.testing.assert_array_equal(slim.n_cells, full.n_cells[rows])
    np.testing.assert_array_equal(slim.stratum_keys, full.stratum_keys[rows])


async def test_read_atomic_rows_empty(strata_store):
    atomic = await strata_store.read_atomic_rows([])
    assert atomic.kind == "atomic"
    # All sliced arrays are empty
    assert atomic.sum_x.size == 0
    assert atomic.sum_xx.size == 0
    assert atomic.nnz.size == 0
    assert atomic.stratum_keys.shape[0] == 0
    assert atomic.n_cells.shape[0] == 0


async def test_read_atomic_rows_no_atomic_raises(no_strata_store):
    import pytest
    with pytest.raises(ValueError, match="no atomic"):
        await no_strata_store.read_atomic_rows([0])


from zarr_access import find_coarse_by_axes, find_coarse_covering


async def test_find_coarse_by_axes_exact_match(strata_store):
    assert find_coarse_by_axes(strata_store, ["cell_type"]) == "cell_type"
    # No coarse covers both axes — only ['cell_type'] exists
    assert find_coarse_by_axes(strata_store, ["cell_type", "donor"]) is None
    assert find_coarse_by_axes(strata_store, ["nope"]) is None


async def test_find_coarse_covering(strata_store):
    # The only coarse table covers ['cell_type'] — request for that returns it
    assert find_coarse_covering(strata_store, ["cell_type"]) == "cell_type"
    # No coarse covers ['donor']
    assert find_coarse_covering(strata_store, ["donor"]) is None
