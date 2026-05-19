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

    def test_uses_configured_chunks_and_shards_for_large_arrays(self, tiny_anndata_zarr: Path):
        """sum_x / sum_xx / nnz must use chunks=(5, n_genes) and shards=(50, n_genes)
        when n_strata is large enough. The tiny_anndata_zarr fixture has a small
        but non-trivial number of strata; the values come from ATOMIC_ROWS_PER_CHUNK
        and ATOMIC_ROWS_PER_SHARD module constants.
        """
        from cell2zarr.strata.io import ATOMIC_ROWS_PER_CHUNK, ATOMIC_ROWS_PER_SHARD

        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        write_atomic(root, atomic, force=False)

        root2 = open_dataset(tiny_anndata_zarr)
        a = root2["uns"]["strata"]["atomic"]
        n_strata, n_genes = atomic.sum_x.shape
        expected_chunk_rows = min(ATOMIC_ROWS_PER_CHUNK, n_strata)
        expected_shard_rows = min(ATOMIC_ROWS_PER_SHARD, n_strata)
        # Shard must be an integer multiple of chunk
        expected_shard_rows = (
            (expected_shard_rows // expected_chunk_rows) * expected_chunk_rows
            or expected_chunk_rows
        )

        for name in ("sum_x", "sum_xx", "nnz"):
            arr = a[name]
            assert arr.chunks == (expected_chunk_rows, n_genes), (
                f"{name}.chunks={arr.chunks}, expected ({expected_chunk_rows}, {n_genes})"
            )
            assert arr.shards == (expected_shard_rows, n_genes), (
                f"{name}.shards={arr.shards}, expected ({expected_shard_rows}, {n_genes})"
            )

    def test_round_trip_arrays_with_new_chunking(self, tiny_anndata_zarr: Path):
        """Round-trip via the existing reader still produces byte-equivalent
        values, even with the new chunk/shard/compressor combination.
        Catches any compressor / shard / encoding regression on read.
        """
        config = StrataConfig(atomic_axes=["cell_type", "donor"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        write_atomic(root, atomic, force=False)

        root2 = open_dataset(tiny_anndata_zarr)
        a = root2["uns"]["strata"]["atomic"]
        np.testing.assert_array_equal(np.asarray(a["sum_x"]), atomic.sum_x)
        np.testing.assert_array_equal(np.asarray(a["sum_xx"]), atomic.sum_xx)
        np.testing.assert_array_equal(np.asarray(a["nnz"]), atomic.nnz)
        np.testing.assert_array_equal(np.asarray(a["n_cells"]), atomic.n_cells)

    def test_chunks_clamp_when_n_strata_smaller_than_chunk_rows(
        self, tiny_anndata_zarr: Path
    ):
        """If n_strata < ATOMIC_ROWS_PER_CHUNK, both chunks and shards clamp
        down to n_strata (no error, no oversized shard).
        """
        from cell2zarr.strata.io import ATOMIC_ROWS_PER_CHUNK

        # Force a tiny atomic by using axes that yield very few non-empty strata.
        # If the tiny fixture's axes don't produce a small enough atomic, this
        # test may need to be skipped or the fixture extended; check n_strata
        # against ATOMIC_ROWS_PER_CHUNK before asserting the clamp.
        config = StrataConfig(atomic_axes=["cell_type"])
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        n_strata, n_genes = atomic.sum_x.shape
        if n_strata >= ATOMIC_ROWS_PER_CHUNK:
            pytest.skip(
                f"tiny fixture has n_strata={n_strata} >= "
                f"ATOMIC_ROWS_PER_CHUNK={ATOMIC_ROWS_PER_CHUNK}; "
                "clamping test needs an even smaller atomic"
            )
        write_atomic(root, atomic, force=False)

        root2 = open_dataset(tiny_anndata_zarr)
        a = root2["uns"]["strata"]["atomic"]
        for name in ("sum_x", "sum_xx", "nnz"):
            arr = a[name]
            # Both should clamp to n_strata
            assert arr.chunks == (n_strata, n_genes)
            assert arr.shards == (n_strata, n_genes)

    def test_coarse_writes_unaffected_by_atomic_chunking(self, tiny_anndata_zarr: Path):
        """Coarse table writes must keep their existing (default) chunk shape —
        the atomic-specific options should not leak to other writers.
        """
        from cell2zarr.strata.io import ATOMIC_ROWS_PER_CHUNK

        config = StrataConfig(
            atomic_axes=["cell_type", "donor"],
            coarse=[["cell_type"]],
        )
        root = open_dataset(tiny_anndata_zarr)
        atomic = build_atomic(root, config)
        coarse = derive_coarse(atomic, ["cell_type"])
        write_atomic(root, atomic, force=False)
        write_coarse(root, "cell_type", coarse, force=False)

        root2 = open_dataset(tiny_anndata_zarr)
        c = root2["uns"]["strata"]["coarse_cell_type"]
        # Coarse should NOT have the atomic-specific chunk shape.
        n_strata_coarse, n_genes = coarse.sum_x.shape
        for name in ("sum_x", "sum_xx", "nnz"):
            arr = c[name]
            assert arr.chunks != (ATOMIC_ROWS_PER_CHUNK, n_genes), (
                f"coarse {name}.chunks={arr.chunks} — atomic chunking leaked into coarse"
            )


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
