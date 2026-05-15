"""End-to-end integration tests using the PBMC3K session fixture."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from cell2zarr.cli import cli
from cell2zarr.strata.io import open_dataset


def _scanpy_groupby_mean(adata: ad.AnnData, column: str) -> tuple[list, np.ndarray]:
    """Compute mean expression per group via plain pandas groupby. Ground truth."""
    X = np.asarray(adata.X, dtype=np.float32)
    df = pd.DataFrame(X)
    df["__group__"] = adata.obs[column].astype(str).values
    means = df.groupby("__group__").mean(numeric_only=True)
    return means.index.tolist(), means.values


class TestIntegrationPbmc3k:
    def test_build_strata_succeeds(self, pbmc3k_zarr: Path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "build-strata",
                str(pbmc3k_zarr),
                "--atomic-axes", "louvain",
                "--coarse", "louvain",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_strata_round_trip_via_anndata(self, pbmc3k_zarr: Path):
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "build-strata",
                str(pbmc3k_zarr),
                "--atomic-axes", "louvain",
                "--coarse", "louvain",
                "--force",
            ],
        )
        root = open_dataset(pbmc3k_zarr)
        assert "strata" in root["uns"]
        assert "atomic" in root["uns"]["strata"]
        assert "coarse_louvain" in root["uns"]["strata"]

    def test_coarse_means_match_scanpy_groupby(self, pbmc3k_zarr: Path):
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "build-strata",
                str(pbmc3k_zarr),
                "--atomic-axes", "louvain",
                "--coarse", "louvain",
                "--force",
            ],
        )
        root = open_dataset(pbmc3k_zarr)
        coarse = root["uns"]["strata"]["coarse_louvain"]
        keys = [row[0] for row in np.asarray(coarse["stratum_keys"])]
        sum_x = np.asarray(coarse["sum_x"])
        n_cells = np.asarray(coarse["n_cells"])
        strata_means = sum_x / n_cells[:, None]

        adata = ad.read_zarr(pbmc3k_zarr)
        expected_groups, expected_means = _scanpy_groupby_mean(adata, "louvain")
        # Align row orders by stratum key
        for stratum_key, row in zip(keys, strata_means):
            expected_row = expected_means[expected_groups.index(stratum_key)]
            np.testing.assert_allclose(row, expected_row, rtol=1e-2, atol=1e-3)


class TestIntegrationErrorPaths:
    def test_no_axes_shows_categoricals(self, pbmc3k_zarr: Path):
        runner = CliRunner()
        result = runner.invoke(cli, ["build-strata", str(pbmc3k_zarr)])
        assert result.exit_code == 2
        assert "Available categorical columns" in result.output
        assert "louvain" in result.output

    def test_force_required_on_rebuild(self, pbmc3k_zarr: Path):
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "build-strata",
                str(pbmc3k_zarr),
                "--atomic-axes", "louvain",
                "--force",
            ],
        )
        result = runner.invoke(
            cli,
            ["build-strata", str(pbmc3k_zarr), "--atomic-axes", "louvain"],
        )
        assert result.exit_code == 1
        assert "already exists" in result.output
