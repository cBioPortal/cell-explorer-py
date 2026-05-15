"""Tests for the top-level build_strata orchestrator."""
from pathlib import Path

import numpy as np
import pytest

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
