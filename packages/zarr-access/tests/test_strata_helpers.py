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


from zarr_access.strata_helpers import strata_variances


def test_strata_variances_basic():
    # stratum 0: 4 cells, gene 0 values [1, 2, 3, 4]
    #   sum_x=10, sum_xx=30, n=4 -> var = (30 - 100/4) / 3 = 5/3 ≈ 1.6667
    table = CoarseStrataTable(
        kind="coarse",
        slug="test",
        axes=["axis"],
        stratum_keys=np.array([["s0"]]),
        gene_indices=None,
        sum_x=np.array([[10.0]], dtype=np.float32),
        sum_xx=np.array([[30.0]], dtype=np.float32),
        nnz=np.array([[4]], dtype=np.int32),
        n_cells=np.array([4], dtype=np.int32),
        schema_version="1.0",
    )
    variances = strata_variances(table)
    np.testing.assert_allclose(variances, [[5 / 3]], rtol=1e-6)


def test_strata_variances_edge_cases():
    # n=0 -> variance=0 (undefined)
    # n=1 -> variance=0 (sample variance is undefined for n=1)
    table0 = make_table(n_strata=1, sum_x_rows=[[0.0]], n_cells=[0])
    table1 = make_table(n_strata=1, sum_x_rows=[[5.0]], n_cells=[1])
    table1 = CoarseStrataTable(
        kind="coarse",
        slug="test",
        axes=["axis"],
        stratum_keys=table1.stratum_keys,
        gene_indices=None,
        sum_x=table1.sum_x,
        sum_xx=np.array([[25.0]], dtype=np.float32),
        nnz=np.array([[1]], dtype=np.int32),
        n_cells=table1.n_cells,
        schema_version="1.0",
    )
    assert strata_variances(table0)[0, 0] == 0.0
    assert strata_variances(table1)[0, 0] == 0.0


from zarr_access.strata_helpers import dotplot_data


def test_dotplot_data_basic():
    # 2 strata × 3 genes, select columns [0, 2]
    table = CoarseStrataTable(
        kind="coarse",
        slug="test",
        axes=["axis"],
        stratum_keys=np.array([["s0"], ["s1"]]),
        gene_indices=None,
        sum_x=np.array([[10.0, 20.0, 30.0], [8.0, 12.0, 16.0]], dtype=np.float32),
        sum_xx=np.zeros((2, 3), dtype=np.float32),
        nnz=np.array([[10, 5, 8], [4, 2, 4]], dtype=np.int32),
        n_cells=np.array([10, 4], dtype=np.int32),
        schema_version="1.0",
    )
    means, frac = dotplot_data(table, [0, 2])
    # means stratum 0: [10/10, 30/10] = [1, 3]
    # means stratum 1: [8/4,   16/4]  = [2, 4]
    np.testing.assert_array_equal(means, np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32))
    # frac stratum 0: [10/10, 8/10] = [1, 0.8]
    # frac stratum 1: [4/4,   4/4]  = [1, 1]
    np.testing.assert_array_equal(frac, np.array([[1.0, 0.8], [1.0, 1.0]], dtype=np.float32))


def test_dotplot_data_empty_gene_indices():
    table = make_table(n_strata=2, sum_x_rows=[[1.0, 2.0], [3.0, 4.0]], n_cells=[1, 1])
    means, frac = dotplot_data(table, [])
    assert means.shape == (2, 0)
    assert frac.shape == (2, 0)
