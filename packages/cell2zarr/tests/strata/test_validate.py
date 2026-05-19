"""Tests for cell2zarr.strata.validate."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell2zarr.strata.io import open_dataset
from cell2zarr.strata.exceptions import StrataPreflightError, StrataConfigError
from cell2zarr.strata.validate import (
    validate_obs_columns,
    estimate_atomic_cardinality,
    list_categorical_obs_columns,
)
from ._v3_fixtures import write_v3_anndata_zarr


@pytest.fixture
def numeric_obs_zarr(tmp_path: Path) -> Path:
    """AnnData with a numeric obs column (not categorical)."""
    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float16)
    obs = pd.DataFrame({
        "n_counts": [100.0, 200.0, 300.0],
        "cell_type": pd.Categorical(["A", "B", "A"]),
    })
    var = pd.DataFrame(index=["gene1"])
    return write_v3_anndata_zarr(tmp_path / "numeric.zarr", X, obs, var)


@pytest.fixture
def nullish_obs_zarr(tmp_path: Path) -> Path:
    """AnnData with one obs column where >5% of cells are missing."""
    n = 100
    X = np.ones((n, 1), dtype=np.float16)
    cats = ["A"] * 90 + [None] * 10  # 10% null
    obs = pd.DataFrame({"sparse_label": pd.Categorical(cats)})
    var = pd.DataFrame(index=["gene1"])
    return write_v3_anndata_zarr(tmp_path / "nullish.zarr", X, obs, var)


class TestValidateObsColumns:
    def test_accepts_categorical_columns(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        # Should not raise
        validate_obs_columns(root, ["cell_type", "donor"])

    def test_rejects_missing_column(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        with pytest.raises(StrataConfigError, match="no_such_column"):
            validate_obs_columns(root, ["cell_type", "no_such_column"])

    def test_rejects_numeric_column(self, numeric_obs_zarr: Path):
        root = open_dataset(numeric_obs_zarr)
        with pytest.raises(StrataConfigError, match="numeric"):
            validate_obs_columns(root, ["n_counts"])

    def test_rejects_nullish_column(self, nullish_obs_zarr: Path):
        root = open_dataset(nullish_obs_zarr)
        with pytest.raises(StrataConfigError, match="missing"):
            validate_obs_columns(root, ["sparse_label"])


class TestEstimateAtomicCardinality:
    def test_counts_unique_tuples(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        # Strata: (A,d1), (A,d2), (B,d1), (B,d2), (C,d1) = 5
        n = estimate_atomic_cardinality(root, ["cell_type", "donor"])
        assert n == 5

    def test_single_axis_unique_count(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        # Strata on cell_type alone: A, B, C = 3
        assert estimate_atomic_cardinality(root, ["cell_type"]) == 3

    def test_raises_when_above_cap(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        with pytest.raises(StrataPreflightError, match="exceeds"):
            estimate_atomic_cardinality(
                root, ["cell_type", "donor"], max_cardinality=3
            )


class TestListCategoricalObsColumns:
    def test_returns_categorical_columns_with_cardinalities(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        cols = list_categorical_obs_columns(root)
        # Sorted dict { name: cardinality }
        assert cols == {"cell_type": 3, "donor": 2}
