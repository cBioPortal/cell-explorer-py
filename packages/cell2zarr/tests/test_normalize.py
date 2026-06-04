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


from cell2zarr._testing import open_zarr, _write_test_h5ad
from cell2zarr.convert import convert_h5ad_to_zarr_chunked


def test_normalize_matches_scanpy(tmp_path):
    import scanpy as sc
    h5ad = tmp_path / "counts.h5ad"
    out = tmp_path / "out.zarr"
    _, adata = _write_counts_h5ad(h5ad, n_obs=200, n_vars=40)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, normalize=True,
        var_chunk_size=10, cell_chunk_size=50,
    )
    convert_h5ad_to_zarr_chunked(cfg)

    X_out = np.asarray(open_zarr(out)["X"][:])

    expected = adata.copy()
    sc.pp.normalize_total(expected)
    sc.pp.log1p(expected)
    X_expected = np.asarray(expected.X)

    np.testing.assert_allclose(X_out, X_expected, rtol=1e-4, atol=1e-4)


def test_normalize_zero_count_cell_stays_zero(tmp_path):
    h5ad = tmp_path / "counts.h5ad"
    out = tmp_path / "out.zarr"
    _write_counts_h5ad(h5ad, n_obs=120, n_vars=30, zero_cell=True)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, normalize=True,
        var_chunk_size=10, cell_chunk_size=50,
    )
    convert_h5ad_to_zarr_chunked(cfg)

    X_out = np.asarray(open_zarr(out)["X"][:])
    assert np.all(np.isfinite(X_out))
    assert np.all(X_out[0, :] == 0.0)


def test_default_path_leaves_X_unchanged(tmp_path):
    h5ad = tmp_path / "raw.h5ad"
    out = tmp_path / "out.zarr"
    _, adata = _write_test_h5ad(h5ad, n_obs=120, n_vars=30)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, normalize=False,
        var_chunk_size=10, cell_chunk_size=50,
    )
    convert_h5ad_to_zarr_chunked(cfg)

    X_out = np.asarray(open_zarr(out)["X"][:])
    np.testing.assert_allclose(X_out, np.asarray(adata.X), rtol=1e-5, atol=1e-5)


def test_normalize_raises_if_all_zero_counts(tmp_path):
    h5ad = tmp_path / "zeros.h5ad"
    out = tmp_path / "out.zarr"
    X = np.zeros((10, 5), dtype=np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(5)])
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(h5ad)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, normalize=True,
        var_chunk_size=5, cell_chunk_size=5,
    )
    with pytest.raises(ValueError, match="all cells have zero total counts"):
        convert_h5ad_to_zarr_chunked(cfg)
