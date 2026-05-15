"""Tests for io read helpers: open_dataset, read_obs_categorical, read_x_block, has_strata."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cell2zarr.strata.io import (
    open_dataset,
    read_obs_categorical,
    read_x_block,
    has_strata,
    existing_strata_summary,
)


class TestOpenDataset:
    def test_returns_zarr_group_with_X_and_obs(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        assert "X" in root
        assert "obs" in root

    def test_writable(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        # We need to be able to write under uns/strata/ later — confirm append-write works
        root.require_group("uns").require_group("strata")
        # Re-open and confirm the group persisted
        root2 = open_dataset(tiny_anndata_zarr)
        assert "strata" in root2["uns"]


class TestReadObsCategorical:
    def test_reads_pandas_categorical(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        ct = read_obs_categorical(root, "cell_type")
        assert isinstance(ct, pd.Categorical)
        assert list(ct) == ["A", "A", "B", "B", "C"]

    def test_missing_column_raises_key_error(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        with pytest.raises(KeyError):
            read_obs_categorical(root, "no_such_column")


class TestReadXBlock:
    def test_returns_float32(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        block = read_x_block(root, slice(0, 2))
        assert block.dtype == np.float32
        assert block.shape == (5, 2)

    def test_values_match_underlying(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        block = read_x_block(root, slice(0, 3))
        # Column 0: [1, 3, 5, 0, 1]
        np.testing.assert_array_equal(block[:, 0], np.array([1, 3, 5, 0, 1], dtype=np.float32))
        # Column 2: [0, 0, 1, 2, 1]
        np.testing.assert_array_equal(block[:, 2], np.array([0, 0, 1, 2, 1], dtype=np.float32))


class TestHasStrata:
    def test_false_on_fresh_dataset(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        assert has_strata(root) is False

    def test_true_after_full_write(self, tiny_anndata_zarr: Path):
        # Manually seed a "fully written" atomic group: arrays + the schema_version marker
        root = open_dataset(tiny_anndata_zarr)
        atomic = root.require_group("uns").require_group("strata").require_group("atomic")
        # Write the schema_version marker LAST. has_strata keys off this.
        atomic.attrs["schema_version"] = "1.0"
        atomic.attrs["axes"] = ["cell_type"]
        atomic.attrs["n_strata"] = 3
        assert has_strata(root) is True

    def test_false_for_partial_write(self, tiny_anndata_zarr: Path):
        """A group exists at uns/strata/atomic/ but has no schema_version — treat as absent."""
        root = open_dataset(tiny_anndata_zarr)
        root.require_group("uns").require_group("strata").require_group("atomic")
        assert has_strata(root) is False


class TestExistingStrataSummary:
    def test_returns_none_when_absent(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        assert existing_strata_summary(root) is None

    def test_returns_attrs_when_present(self, tiny_anndata_zarr: Path):
        root = open_dataset(tiny_anndata_zarr)
        atomic = root.require_group("uns").require_group("strata").require_group("atomic")
        atomic.attrs["schema_version"] = "1.0"
        atomic.attrs["axes"] = ["cell_type"]
        atomic.attrs["n_strata"] = 3
        summary = existing_strata_summary(root)
        assert summary is not None
        assert summary["axes"] == ["cell_type"]
        assert summary["n_strata"] == 3
