"""Tests for derive_coarse — axis-collapse from atomic."""
from pathlib import Path

import numpy as np

from cell2zarr.strata.config import StrataConfig
from cell2zarr.strata.io import open_dataset
from cell2zarr.strata.build import build_atomic
from cell2zarr.strata.derive import derive_coarse, CoarseTable


class TestDeriveCoarseIdentity:
    def test_keeping_all_axes_equals_atomic(self, tiny_anndata_zarr: Path):
        """Coarse keeping ALL atomic axes is identical to atomic (just renamed)."""
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)

        coarse = derive_coarse(atomic, ["cell_type", "donor"])

        assert isinstance(coarse, CoarseTable)
        assert coarse.axis_names == ["cell_type", "donor"]
        # Same n_strata, same arrays (sorted lexicographically by axis values)
        assert coarse.sum_x.shape == atomic.sum_x.shape

        # Build a sorted view of the atomic table to compare against the
        # deterministically-sorted coarse output.
        order = np.lexsort(atomic.stratum_keys.T[::-1])
        np.testing.assert_array_equal(coarse.stratum_keys, atomic.stratum_keys[order])
        np.testing.assert_allclose(coarse.sum_x, atomic.sum_x[order])
        np.testing.assert_allclose(coarse.sum_xx, atomic.sum_xx[order])
        np.testing.assert_array_equal(coarse.nnz, atomic.nnz[order])
        np.testing.assert_array_equal(coarse.n_cells, atomic.n_cells[order])

    def test_partial_collapse_per_cell_type(self, tiny_anndata_zarr: Path):
        """coarse cell_type should equal: sum over donor of atomic (cell_type, donor)."""
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)

        coarse = derive_coarse(atomic, ["cell_type"])
        assert coarse.axis_names == ["cell_type"]

        # For each cell_type in coarse, verify its sum_x is the sum of all
        # matching atomic rows.
        for ct_idx, ct in enumerate(coarse.stratum_keys[:, 0]):
            atomic_match = atomic.stratum_keys[:, 0] == ct
            expected_sum_x = atomic.sum_x[atomic_match].sum(axis=0)
            np.testing.assert_allclose(coarse.sum_x[ct_idx], expected_sum_x)

    def test_full_collapse_to_global(self, tiny_anndata_zarr: Path):
        """coarse with [] axes is a single-row table holding dataset-wide sums."""
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)

        coarse = derive_coarse(atomic, [])
        assert coarse.sum_x.shape == (1, 3)
        np.testing.assert_allclose(coarse.sum_x[0], atomic.sum_x.sum(axis=0))
        assert int(coarse.n_cells[0]) == int(atomic.n_cells.sum())


class TestDeriveCoarseDeterministic:
    def test_deterministic_row_order_across_runs(self, tiny_anndata_zarr: Path):
        """Same atomic + same axes → byte-identical coarse output."""
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)

        c1 = derive_coarse(atomic, ["cell_type"])
        c2 = derive_coarse(atomic, ["cell_type"])

        np.testing.assert_array_equal(c1.stratum_keys, c2.stratum_keys)
        np.testing.assert_array_equal(c1.sum_x, c2.sum_x)

    def test_lexicographic_sort(self, tiny_anndata_zarr: Path):
        """coarse.stratum_keys is lexicographically sorted on the kept axes."""
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)

        coarse = derive_coarse(atomic, ["donor", "cell_type"])
        # Sorted by donor first, then cell_type
        rows = coarse.stratum_keys.tolist()
        assert rows == sorted(rows)
