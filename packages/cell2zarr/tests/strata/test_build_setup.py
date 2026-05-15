"""Tests for build.py setup-phase helpers."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell2zarr.strata.io import open_dataset
from cell2zarr.strata.build import (
    AtomicTable,
    compute_strata_mapping,
)


class TestComputeStrataMapping:
    def test_two_axes_five_cells(self, tiny_anndata_zarr: Path):
        """Tiny fixture: (cell_type, donor) → 5 unique strata in order of first appearance."""
        root = open_dataset(tiny_anndata_zarr)
        stratum_keys, cell_to_stratum, n_cells = compute_strata_mapping(
            root, ["cell_type", "donor"]
        )

        # 5 unique strata. The ordering is implementation-defined but the
        # tiny-fixture cells map to 5 distinct stratum ids.
        assert len(stratum_keys) == 5
        assert stratum_keys.shape == (5, 2)
        assert cell_to_stratum.shape == (5,)
        assert cell_to_stratum.dtype == np.int32
        assert set(cell_to_stratum) == {0, 1, 2, 3, 4}

        # n_cells is one per stratum since each tuple is unique to one cell
        assert (n_cells == 1).all()
        assert n_cells.dtype == np.int32

    def test_single_axis_groups_cells(self, tiny_anndata_zarr: Path):
        """cell_type alone: A (2 cells), B (2 cells), C (1 cell) → 3 strata."""
        root = open_dataset(tiny_anndata_zarr)
        stratum_keys, cell_to_stratum, n_cells = compute_strata_mapping(
            root, ["cell_type"]
        )
        assert len(stratum_keys) == 3
        assert n_cells.sum() == 5
        # Cell counts: 2 (A) + 2 (B) + 1 (C)
        assert sorted(n_cells.tolist()) == [1, 2, 2]

    def test_stratum_keys_preserve_axis_order(self, tiny_anndata_zarr: Path):
        """stratum_keys columns are in axis_names order."""
        root = open_dataset(tiny_anndata_zarr)
        stratum_keys, _, _ = compute_strata_mapping(root, ["donor", "cell_type"])
        # First column should be donors (d1 or d2), second cell types (A, B, or C)
        donors = set(stratum_keys[:, 0])
        cell_types = set(stratum_keys[:, 1])
        assert donors <= {"d1", "d2"}
        assert cell_types <= {"A", "B", "C"}


class TestStratumKeyWidth:
    """Stratum keys must preserve full string content — no truncation regardless of length."""

    @pytest.fixture
    def long_label_zarr(self, tmp_path: Path) -> Path:
        # 40 chars — longer than the old hardcoded <U32 limit.
        long_name = "fallopian tube secretory epithelial cell"
        X = np.array([[1.0], [2.0]], dtype=np.float16)
        obs = pd.DataFrame({
            "cell_type": pd.Categorical([long_name, "B cell"]),
        })
        var = pd.DataFrame(index=["gene1"])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        path = tmp_path / "long.zarr"
        adata.write_zarr(path)
        return path

    def test_long_label_not_truncated(self, long_label_zarr: Path):
        root = open_dataset(long_label_zarr)
        stratum_keys, _, _ = compute_strata_mapping(root, ["cell_type"])
        labels = set(stratum_keys[:, 0].tolist())
        assert "fallopian tube secretory epithelial cell" in labels, (
            f"long label was truncated; got {labels}"
        )
        assert "B cell" in labels

    def test_stratum_keys_dtype_width_sized_to_max(self, long_label_zarr: Path):
        root = open_dataset(long_label_zarr)
        stratum_keys, _, _ = compute_strata_mapping(root, ["cell_type"])
        # Dtype is fixed-width unicode wide enough for the longest label.
        assert stratum_keys.dtype.kind == "U"
        assert stratum_keys.dtype.itemsize // 4 >= 40  # U-dtype itemsize is bytes (4 per char in UCS-4)


class TestAtomicTableShape:
    def test_dataclass_fields(self):
        table = AtomicTable(
            axis_names=["cell_type"],
            stratum_keys=np.array([["A"], ["B"]]),
            sum_x=np.zeros((2, 3), dtype=np.float32),
            sum_xx=np.zeros((2, 3), dtype=np.float32),
            nnz=np.zeros((2, 3), dtype=np.int32),
            n_cells=np.zeros(2, dtype=np.int32),
        )
        assert table.axis_names == ["cell_type"]
        assert table.sum_x.shape == (2, 3)
        assert table.n_cells.shape == (2,)
