"""Tests for the top-level build_strata orchestrator."""
from pathlib import Path

import numpy as np
import pytest
import zarr

from cell2zarr.strata import build_strata
from cell2zarr.strata.config import StrataConfig
from cell2zarr.strata.exceptions import StrataConfigError, StrataExistsError
from cell2zarr.strata.io import open_dataset


class TestBuildStrataHappyPath:
    def test_writes_atomic_only(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type"])
        build_strata(tiny_anndata_zarr, config)

        root = open_dataset(tiny_anndata_zarr)
        assert "atomic" in root["uns"]["strata"]
        assert "schema_version" in root["uns"]["strata"]["atomic"].attrs

    def test_writes_atomic_and_one_coarse(self, tiny_anndata_zarr: Path):
        config = StrataConfig(
            atomic_axes=["cell_type", "donor"],
            coarse=[["cell_type"]],
        )
        build_strata(tiny_anndata_zarr, config)

        root = open_dataset(tiny_anndata_zarr)
        strata = root["uns"]["strata"]
        assert "atomic" in strata
        assert "coarse_cell_type" in strata

    def test_writes_atomic_and_multi_coarse(self, tiny_anndata_zarr: Path):
        config = StrataConfig(
            atomic_axes=["cell_type", "donor"],
            coarse=[["cell_type"], ["donor"]],
        )
        build_strata(tiny_anndata_zarr, config)

        root = open_dataset(tiny_anndata_zarr)
        strata = root["uns"]["strata"]
        assert "coarse_cell_type" in strata
        assert "coarse_donor" in strata


class TestBuildStrataValidation:
    def test_raises_on_missing_obs_column(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["no_such_column"])
        with pytest.raises(StrataConfigError):
            build_strata(tiny_anndata_zarr, config)

    def test_raises_on_existing_strata_without_force(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type"])
        build_strata(tiny_anndata_zarr, config)

        with pytest.raises(StrataExistsError):
            build_strata(tiny_anndata_zarr, config)

    def test_force_overwrites_existing(self, tiny_anndata_zarr: Path):
        config = StrataConfig(atomic_axes=["cell_type"])
        build_strata(tiny_anndata_zarr, config)

        config_force = StrataConfig(atomic_axes=["donor"], force=True)
        build_strata(tiny_anndata_zarr, config_force)

        root = open_dataset(tiny_anndata_zarr)
        assert list(root["uns"]["strata"]["atomic"].attrs["axes"]) == ["donor"]


class TestBuildStrataAddCoarseOnly:
    """`--add-coarse-only` reuses the existing atomic instead of re-reading X."""

    def test_adds_coarse_without_rebuilding_atomic(self, tiny_anndata_zarr: Path):
        # Initial build.
        build_strata(
            tiny_anndata_zarr,
            StrataConfig(
                atomic_axes=["cell_type", "donor"],
                coarse=[["cell_type"]],
            ),
        )
        root = open_dataset(tiny_anndata_zarr)
        atomic_before = {
            "axes": list(root["uns"]["strata"]["atomic"].attrs["axes"]),
            "sum_x": np.asarray(root["uns"]["strata"]["atomic"]["sum_x"]).copy(),
        }

        # Add a new coarse table without --force; atomic must be untouched.
        build_strata(
            tiny_anndata_zarr,
            StrataConfig(
                atomic_axes=["cell_type", "donor"],
                coarse=[["donor"]],
                add_coarse_only=True,
            ),
        )

        root = open_dataset(tiny_anndata_zarr)
        # New coarse is present, original coarse still present.
        assert "coarse_donor" in root["uns"]["strata"]
        assert "coarse_cell_type" in root["uns"]["strata"]
        # Atomic was not rebuilt.
        atomic_after = root["uns"]["strata"]["atomic"]
        assert list(atomic_after.attrs["axes"]) == atomic_before["axes"]
        np.testing.assert_array_equal(np.asarray(atomic_after["sum_x"]), atomic_before["sum_x"])

    def test_add_coarse_only_requires_existing_atomic(self, tiny_anndata_zarr: Path):
        # No prior build; add-coarse-only must fail rather than silently rebuilding.
        with pytest.raises(StrataExistsError, match="no atomic"):
            build_strata(
                tiny_anndata_zarr,
                StrataConfig(
                    coarse=[["cell_type"]],
                    add_coarse_only=True,
                ),
            )

    def test_add_coarse_only_rejects_unknown_axis(self, tiny_anndata_zarr: Path):
        # Existing atomic has only cell_type; deriving a coarse on donor is impossible.
        build_strata(
            tiny_anndata_zarr,
            StrataConfig(atomic_axes=["cell_type"]),
        )
        with pytest.raises(StrataConfigError, match="donor"):
            build_strata(
                tiny_anndata_zarr,
                StrataConfig(
                    coarse=[["donor"]],
                    add_coarse_only=True,
                ),
            )

    def test_add_coarse_only_works_without_atomic_axes_in_config(
        self, tiny_anndata_zarr: Path,
    ):
        # User shouldn't need to re-state atomic axes; the on-disk atomic carries them.
        build_strata(
            tiny_anndata_zarr,
            StrataConfig(atomic_axes=["cell_type", "donor"]),
        )
        build_strata(
            tiny_anndata_zarr,
            StrataConfig(coarse=[["cell_type"]], add_coarse_only=True),
        )
        root = open_dataset(tiny_anndata_zarr)
        assert "coarse_cell_type" in root["uns"]["strata"]


class TestBuildStrataConsolidation:
    """After build_strata finishes, the consolidated metadata index must reflect
    the new uns/strata/* groups so production readers (which open with
    use_consolidated=True by default) can see them."""

    def test_strata_visible_via_consolidated_metadata(self, tiny_anndata_zarr: Path):
        config = StrataConfig(
            atomic_axes=["cell_type", "donor"],
            coarse=[["cell_type"]],
        )
        build_strata(tiny_anndata_zarr, config)

        # Production-style read: default consolidation enabled.
        root = zarr.open_group(str(tiny_anndata_zarr), mode="r")
        assert "strata" in root["uns"], (
            "uns/strata not visible via consolidated metadata; "
            "build_strata must re-consolidate after writes"
        )
        assert "atomic" in root["uns"]["strata"]
        assert "coarse_cell_type" in root["uns"]["strata"]
        # Round-trip a payload to confirm arrays are addressable too.
        atomic = root["uns"]["strata"]["atomic"]
        assert atomic.attrs["schema_version"] == "1.0"
        assert int(atomic.attrs["n_strata"]) == len(np.asarray(atomic["n_cells"]))

    def test_force_rebuild_keeps_consolidated_in_sync(self, tiny_anndata_zarr: Path):
        # First build with cell_type only.
        build_strata(tiny_anndata_zarr, StrataConfig(atomic_axes=["cell_type"]))
        # Force-rebuild with different axes + coarse.
        build_strata(
            tiny_anndata_zarr,
            StrataConfig(
                atomic_axes=["donor"],
                coarse=[["donor"]],
                force=True,
            ),
        )

        root = zarr.open_group(str(tiny_anndata_zarr), mode="r")
        assert list(root["uns"]["strata"]["atomic"].attrs["axes"]) == ["donor"]
        assert "coarse_donor" in root["uns"]["strata"]
