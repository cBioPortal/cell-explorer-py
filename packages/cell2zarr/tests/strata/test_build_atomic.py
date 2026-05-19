"""Correctness tests for build_atomic.

The math is verified against a per-cell numpy ground truth on the tiny fixture.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell2zarr.strata.config import StrataConfig
from cell2zarr.strata.io import open_dataset
from cell2zarr.strata.build import build_atomic, compute_strata_mapping
from ._v3_fixtures import write_v3_anndata_zarr


def _naive_atomic(
    X_f32: np.ndarray,
    cell_to_stratum: np.ndarray,
    n_strata: int,
):
    """Reference implementation: per-cell loop. Used as ground truth."""
    n_genes = X_f32.shape[1]
    sum_x = np.zeros((n_strata, n_genes), dtype=np.float32)
    sum_xx = np.zeros((n_strata, n_genes), dtype=np.float32)
    nnz = np.zeros((n_strata, n_genes), dtype=np.int32)
    n_cells = np.zeros(n_strata, dtype=np.int32)
    for cell_idx in range(X_f32.shape[0]):
        s = int(cell_to_stratum[cell_idx])
        row = X_f32[cell_idx]
        sum_x[s] += row
        sum_xx[s] += row * row
        nnz[s] += (row > 0).astype(np.int32)
        n_cells[s] += 1
    return sum_x, sum_xx, nnz, n_cells


@pytest.fixture
def large_value_anndata_zarr(tmp_path: Path) -> Path:
    """5 cells × 1 gene with a value above sqrt(65504) ≈ 256.

    Squaring this value in float16 would overflow to inf. Verifies the upcast.
    """
    X = np.array([[300.0], [400.0], [500.0], [0.0], [100.0]], dtype=np.float16)
    obs = pd.DataFrame({"cell_type": pd.Categorical(["A", "A", "B", "B", "C"])})
    var = pd.DataFrame(index=["gene1"])
    return write_v3_anndata_zarr(tmp_path / "large.zarr", X, obs, var)


class TestBuildAtomicCorrectness:
    def test_two_axes_matches_naive_numpy(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)

        # Build the ground truth using float32 X and the same stratum mapping.
        X_f32 = np.array(
            [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 0.0],
                [5.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        stratum_keys, cell_to_stratum, _ = compute_strata_mapping(
            root, ["cell_type", "donor"]
        )
        expected_sum_x, expected_sum_xx, expected_nnz, expected_n_cells = _naive_atomic(
            X_f32, cell_to_stratum, n_strata=len(stratum_keys)
        )

        np.testing.assert_allclose(atomic.sum_x, expected_sum_x, rtol=1e-5)
        np.testing.assert_allclose(atomic.sum_xx, expected_sum_xx, rtol=1e-5)
        np.testing.assert_array_equal(atomic.nnz, expected_nnz)
        np.testing.assert_array_equal(atomic.n_cells, expected_n_cells)

    def test_single_axis(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)

        # 3 strata (A, B, C). Cell counts: 2 + 2 + 1.
        assert atomic.sum_x.shape == (3, 3)
        assert atomic.sum_x.dtype == np.float32
        assert atomic.n_cells.sum() == 5

    def test_no_overflow_on_large_values(self, large_value_anndata_zarr: Path):
        """sum_xx must stay finite even with cell values > sqrt(65504)."""
        config = StrataConfig(atomic_axes=["cell_type"])
        root = open_dataset(large_value_anndata_zarr)
        atomic = build_atomic(root, config)

        assert np.isfinite(atomic.sum_x).all()
        assert np.isfinite(atomic.sum_xx).all()
        # Stratum A: cells [300, 400] → sum_xx = 90000 + 160000 = 250000
        # Find stratum A by looking at stratum_keys
        a_idx = int(np.where(atomic.stratum_keys[:, 0] == "A")[0][0])
        np.testing.assert_allclose(atomic.sum_xx[a_idx, 0], 250000.0, rtol=1e-3)


class TestBuildAtomicShape:
    def test_returns_axis_names_in_config_order(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["donor", "cell_type"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        assert atomic.axis_names == ["donor", "cell_type"]
