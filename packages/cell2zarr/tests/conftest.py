"""Shared test fixtures for cell2zarr."""
import numpy as np
import pandas as pd
import anndata as ad
import zarr
import pytest


@pytest.fixture
def h5ad_path(tmp_path):
    return tmp_path / "test.h5ad"


@pytest.fixture
def zarr_path(tmp_path):
    return tmp_path / "test.zarr"


@pytest.fixture
def sample_h5ad(h5ad_path):
    """Create a minimal h5ad with obs, var, obsm, uns. Returns (path, adata)."""
    return _write_test_h5ad(h5ad_path)


@pytest.fixture
def sample_store(zarr_path):
    """Create a minimal empty zarr v3 store. Returns path."""
    _create_zarr_store(zarr_path)
    return zarr_path


def _write_test_h5ad(path, n_obs=100, n_vars=50):
    """Create a minimal h5ad file. Returns (path, adata)."""
    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": ["A"] * (n_obs // 2) + ["B"] * (n_obs // 2)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = np.random.randn(n_obs, 2).astype(np.float32)
    adata.uns["method"] = "test"
    adata.write_h5ad(path)
    return path, adata


def _create_zarr_store(path):
    """Create a minimal empty zarr v3 store."""
    store = zarr.storage.LocalStore(str(path))
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.attrs["encoding-type"] = "anndata"
    root.attrs["encoding-version"] = "0.1.0"


def open_zarr(path):
    """Open a zarr store for reading."""
    return zarr.open_group(zarr.storage.LocalStore(str(path)), mode="r")
