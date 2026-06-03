"""Tests for the --normalize (scanpy normalize_total + log1p) option."""
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from cell2zarr.models import ConversionConfig


def _write_counts_h5ad(path, n_obs=200, n_vars=40, seed=0, zero_cell=False):
    """Write an h5ad of integer raw counts. Returns (path, adata)."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    if zero_cell:
        X[0, :] = 0.0
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)
    return path, adata


def test_conversionconfig_normalize_defaults_false(tmp_path):
    cfg = ConversionConfig(input_file=tmp_path / "in.h5ad", output_file=tmp_path / "out.zarr")
    assert cfg.normalize is False


def test_conversionconfig_normalize_can_be_set(tmp_path):
    cfg = ConversionConfig(
        input_file=tmp_path / "in.h5ad", output_file=tmp_path / "out.zarr", normalize=True
    )
    assert cfg.normalize is True
