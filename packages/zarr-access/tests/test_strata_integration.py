"""End-to-end integration tests for StrataStore.

Reads the strata-tiny.zarr fixture via the HTTP fixture_server and asserts
ground-truth correctness. Same pattern as test_anndata_store.py.
"""

import numpy as np
import pytest_asyncio

from zarr_access import StrataStore, ZarrStore, strata_means


@pytest_asyncio.fixture
async def strata_store(fixture_server):
    zarr_store = await ZarrStore.open(f"{fixture_server}/strata-tiny.zarr")
    return await StrataStore.open(zarr_store)


async def test_coarse_cell_type_ground_truth_cell_type_A_gene_0(strata_store):
    """Fixture cells 0..49 with X[i, j] = ((i+1) * (j+1)) % 5.

    cell_type=A covers cells 0..19 (20 cells).
    For gene 0 (j=0), values are (i+1) % 5 for i in 0..19:
        repeats 4 times of [1,2,3,4,0]
        sum = 4 * (1+2+3+4+0) = 40
    """
    coarse = await strata_store.read_coarse("cell_type")
    # Find stratum A
    a_rows = np.where(coarse.stratum_keys[:, 0] == "A")[0]
    assert len(a_rows) == 1
    a_idx = int(a_rows[0])
    assert int(coarse.n_cells[a_idx]) == 20
    # Float16 -> float32 round-trip should produce exactly 40
    assert float(coarse.sum_x[a_idx, 0]) == 40.0


async def test_atomic_cell_counts_sum_to_50(strata_store):
    atomic = await strata_store.read_atomic()
    assert int(atomic.n_cells.sum()) == 50


async def test_helpers_compose_on_fixture_data(strata_store):
    """The pure helpers compose correctly on real fixture data."""
    coarse = await strata_store.read_coarse("cell_type")
    means = strata_means(coarse)
    # Means should be finite and at least one cell is non-zero
    assert np.isfinite(means).all()
    assert (means > 0).any()
