"""Tests for io write primitives."""
from pathlib import Path

import numpy as np
import pytest

from cell2zarr.strata.config import StrataConfig
from cell2zarr.strata.exceptions import StrataExistsError
from cell2zarr.strata.io import (
    open_dataset,
    has_strata,
    write_atomic,
    write_coarse,
)
from cell2zarr.strata.build import build_atomic
from cell2zarr.strata.derive import derive_coarse


class TestWriteAtomic:
    def test_round_trip_arrays(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        write_atomic(root, atomic, force=False)

        # Re-open and inspect the persisted state
        root2 = open_dataset(tiny_anndata_zarr)
        a = root2["uns"]["strata"]["atomic"]
        assert a.attrs["schema_version"] == "1.0"
        assert list(a.attrs["axes"]) == ["cell_type", "donor"]
        np.testing.assert_allclose(np.asarray(a["sum_x"]), atomic.sum_x)
        np.testing.assert_allclose(np.asarray(a["sum_xx"]), atomic.sum_xx)
        np.testing.assert_array_equal(np.asarray(a["nnz"]), atomic.nnz)
        np.testing.assert_array_equal(np.asarray(a["n_cells"]), atomic.n_cells)
        assert has_strata(root2) is True

    def test_existing_atomic_raises_without_force(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        write_atomic(root, atomic, force=False)

        with pytest.raises(StrataExistsError, match="already exists"):
            write_atomic(root, atomic, force=False)

    def test_force_overwrites(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        write_atomic(root, atomic, force=False)

        # Rebuild with different axes and force-overwrite
        config2 = StrataConfig(atomic_axes=["donor"])
        atomic2 = build_atomic(root, config2)
        write_atomic(root, atomic2, force=True)

        root2 = open_dataset(tiny_anndata_zarr)
        assert list(root2["uns"]["strata"]["atomic"].attrs["axes"]) == ["donor"]


class TestWriteCoarse:
    def test_round_trip_arrays(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        write_atomic(root, atomic, force=False)
        coarse = derive_coarse(atomic, ["cell_type"])
        write_coarse(root, "cell_type", coarse, force=False)

        root2 = open_dataset(tiny_anndata_zarr)
        c = root2["uns"]["strata"]["coarse_cell_type"]
        assert c.attrs["schema_version"] == "1.0"
        assert list(c.attrs["axes"]) == ["cell_type"]
        assert c.attrs["derived_from"] == "atomic"
        np.testing.assert_allclose(np.asarray(c["sum_x"]), coarse.sum_x)


class TestPartialBuildDetection:
    def test_partial_write_not_visible_to_has_strata(self, tiny_anndata_zarr: Path):
        """A half-finished atomic (group exists, no schema_version) is invisible."""
        root = open_dataset(tiny_anndata_zarr)
        root.require_group("uns").require_group("strata").require_group("atomic")
        # No schema_version attribute → treated as absent
        assert has_strata(root) is False

    def test_force_recovers_from_partial_write(self, tiny_anndata_zarr: Path):
        """If a previous build was killed, --force lets the next build clobber the husk."""
        config = StrataConfig(atomic_axes=["cell_type"])
        root = open_dataset(tiny_anndata_zarr)
        # Simulate partial build
        root.require_group("uns").require_group("strata").require_group("atomic")
        atomic = build_atomic(root, config)
        # Without force we'd error if has_strata=True, but partial husk has has_strata=False;
        # however the group EXISTS, and write_atomic must replace it.
        write_atomic(root, atomic, force=True)
        assert has_strata(root) is True
