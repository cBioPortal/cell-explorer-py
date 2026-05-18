"""Pure unit tests for strata_helpers — no fixture, no network."""

import numpy as np

from zarr_access.strata_helpers import strata_means
from zarr_access.strata_store import CoarseStrataTable


def make_table(
    n_strata: int,
    sum_x_rows: list[list[float]],
    n_cells: list[int],
) -> CoarseStrataTable:
    sum_x = np.array(sum_x_rows, dtype=np.float32)
    return CoarseStrataTable(
        kind="coarse",
        slug="test",
        axes=["axis"],
        stratum_keys=np.array([[f"s{i}"] for i in range(n_strata)]),
        gene_indices=None,
        sum_x=sum_x,
        sum_xx=np.zeros_like(sum_x),
        nnz=np.zeros(sum_x.shape, dtype=np.int32),
        n_cells=np.array(n_cells, dtype=np.int32),
        schema_version="1.0",
    )


def test_strata_means_basic():
    # stratum 0: n_cells=10, sums=[10, 20, 30] -> means=[1, 2, 3]
    # stratum 1: n_cells=4,  sums=[8,  12, 16] -> means=[2, 3, 4]
    table = make_table(
        n_strata=2,
        sum_x_rows=[[10.0, 20.0, 30.0], [8.0, 12.0, 16.0]],
        n_cells=[10, 4],
    )
    means = strata_means(table)
    np.testing.assert_array_equal(means, np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32))


def test_strata_means_zero_cells():
    # n_cells=0 -> means=0 (no NaN propagation)
    table = make_table(n_strata=1, sum_x_rows=[[0.0, 0.0]], n_cells=[0])
    means = strata_means(table)
    np.testing.assert_array_equal(means, np.zeros((1, 2), dtype=np.float32))


from zarr_access.strata_helpers import strata_frac_expressing


def test_strata_frac_expressing_basic():
    # stratum 0: n_cells=10, nnz=[5, 0, 10]   -> frac=[0.5, 0, 1]
    # stratum 1: n_cells=4,  nnz=[1, 2, 4]    -> frac=[0.25, 0.5, 1]
    sum_x = np.zeros((2, 3), dtype=np.float32)
    table = CoarseStrataTable(
        kind="coarse",
        slug="test",
        axes=["axis"],
        stratum_keys=np.array([["s0"], ["s1"]]),
        gene_indices=None,
        sum_x=sum_x,
        sum_xx=sum_x.copy(),
        nnz=np.array([[5, 0, 10], [1, 2, 4]], dtype=np.int32),
        n_cells=np.array([10, 4], dtype=np.int32),
        schema_version="1.0",
    )
    frac = strata_frac_expressing(table)
    np.testing.assert_array_equal(
        frac,
        np.array([[0.5, 0.0, 1.0], [0.25, 0.5, 1.0]], dtype=np.float32),
    )


def test_strata_frac_expressing_zero_cells():
    table = make_table(n_strata=1, sum_x_rows=[[0.0, 0.0]], n_cells=[0])
    table = CoarseStrataTable(
        kind="coarse",
        slug="test",
        axes=["axis"],
        stratum_keys=table.stratum_keys,
        gene_indices=None,
        sum_x=table.sum_x,
        sum_xx=table.sum_xx,
        nnz=np.zeros((1, 2), dtype=np.int32),
        n_cells=table.n_cells,
        schema_version=table.schema_version,
    )
    frac = strata_frac_expressing(table)
    np.testing.assert_array_equal(frac, np.zeros((1, 2), dtype=np.float32))
