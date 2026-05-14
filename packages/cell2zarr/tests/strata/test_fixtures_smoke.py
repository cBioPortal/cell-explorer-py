"""Sanity-check that the shared fixtures load and look right."""
from pathlib import Path

import anndata as ad
import numpy as np


class TestTinyAnndataZarr:
    def test_path_exists(self, tiny_anndata_zarr: Path):
        assert tiny_anndata_zarr.exists()
        assert tiny_anndata_zarr.suffix == ".zarr"

    def test_round_trips_through_anndata(self, tiny_anndata_zarr: Path):
        adata = ad.read_zarr(tiny_anndata_zarr)
        assert adata.shape == (5, 3)
        assert adata.X.dtype == np.float16
        assert "cell_type" in adata.obs.columns
        assert "donor" in adata.obs.columns
        assert list(adata.obs["cell_type"].cat.categories) == ["A", "B", "C"]
