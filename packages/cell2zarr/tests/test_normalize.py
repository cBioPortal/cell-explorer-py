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


from anndata.io import read_elem


def test_normalize_writes_log1p_uns(tmp_path):
    h5ad = tmp_path / "counts.h5ad"
    out = tmp_path / "out.zarr"
    _write_counts_h5ad(h5ad, n_obs=100, n_vars=20)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, normalize=True,
        var_chunk_size=10, cell_chunk_size=50,
    )
    convert_h5ad_to_zarr_chunked(cfg)

    root = open_zarr(out)
    uns = read_elem(root["uns"])
    assert uns["log1p"] == {"base": None}


def test_default_path_has_no_log1p_uns(tmp_path):
    h5ad = tmp_path / "raw.h5ad"
    out = tmp_path / "out.zarr"
    _write_test_h5ad(h5ad, n_obs=100, n_vars=20)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, normalize=False,
        var_chunk_size=10, cell_chunk_size=50,
    )
    convert_h5ad_to_zarr_chunked(cfg)

    root = open_zarr(out)
    uns = read_elem(root["uns"]) if "uns" in root else {}
    assert "log1p" not in uns


from click.testing import CliRunner
from cell2zarr.cli import cli


def test_cli_normalize_flag_log_normalizes_X(tmp_path):
    import scanpy as sc
    h5ad = tmp_path / "counts.h5ad"
    out = tmp_path / "out.zarr"
    _, adata = _write_counts_h5ad(h5ad, n_obs=120, n_vars=20)

    result = CliRunner().invoke(
        cli, ["convert", str(h5ad), str(out), "--two-phase", "--normalize",
              "--cell-chunk-size", "50", "--var-chunk-size", "10"]
    )
    assert result.exit_code == 0, result.output

    X_out = np.asarray(open_zarr(out)["X"][:])
    expected = adata.copy()
    sc.pp.normalize_total(expected)
    sc.pp.log1p(expected)
    np.testing.assert_allclose(X_out, np.asarray(expected.X), rtol=1e-4, atol=1e-4)


def test_cli_normalize_without_two_phase_warns_and_ignores(tmp_path):
    h5ad = tmp_path / "counts.h5ad"
    out = tmp_path / "out.zarr"
    _, adata = _write_counts_h5ad(h5ad, n_obs=60, n_vars=15)

    result = CliRunner().invoke(cli, ["convert", str(h5ad), str(out), "--normalize"])
    assert result.exit_code == 0, result.output
    assert "--normalize requires --two-phase" in result.output

    # Simple path does not normalize: X stays equal to the raw input.
    adata_out = ad.read_zarr(out)
    X_out = np.asarray(adata_out.X.toarray() if hasattr(adata_out.X, 'toarray') else adata_out.X)
    X_expected = np.asarray(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    np.testing.assert_allclose(X_out, X_expected, rtol=1e-5, atol=1e-5)


def test_normalize_with_n_top_genes_uses_full_gene_totals(tmp_path):
    import scanpy as sc
    from anndata.io import read_elem
    h5ad = tmp_path / "hvg.h5ad"
    out = tmp_path / "out.zarr"
    rng = np.random.default_rng(2)
    n_obs, n_vars = 200, 40
    X = rng.poisson(3.0, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    var["highly_variable"] = [True] * 15 + [False] * (n_vars - 15)
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(h5ad)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, normalize=True, n_top_genes=10,
        dtype="float64", var_chunk_size=5, cell_chunk_size=50,
    )
    convert_h5ad_to_zarr_chunked(cfg)

    root = open_zarr(out)
    X_out = np.asarray(root["X"][:])
    selected = list(read_elem(root["var"]).index)  # genes the pipeline kept, in output order

    # scanpy oracle: normalize on the FULL gene set, then subset to the same genes.
    # Cast X to float64 so scanpy's arithmetic matches the pipeline's float64 math;
    # without this cast scanpy stays in float32, diverging ~1e-7 from our float64 output.
    expected = ad.read_h5ad(h5ad)
    expected.X = np.asarray(expected.X, dtype=np.float64)
    sc.pp.normalize_total(expected)
    sc.pp.log1p(expected)
    expected_sub = expected[:, selected]

    assert X_out.shape == (n_obs, 10)
    np.testing.assert_allclose(X_out, np.asarray(expected_sub.X), rtol=1e-9, atol=1e-9)


def test_normalize_float64_precision_matches_scanpy(tmp_path):
    import scanpy as sc
    h5ad = tmp_path / "f64.h5ad"
    out = tmp_path / "out.zarr"
    rng = np.random.default_rng(1)
    # Non-integer float64 values that are NOT exactly representable in float32,
    # so rounding the raw to float32 before the math would visibly diverge.
    X = (rng.random((150, 25)) * 1000.0).astype(np.float64)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(150)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(25)])
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(h5ad)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, normalize=True,
        dtype="float64", var_chunk_size=10, cell_chunk_size=50,
    )
    convert_h5ad_to_zarr_chunked(cfg)
    X_out = np.asarray(open_zarr(out)["X"][:])

    expected = ad.read_h5ad(h5ad)
    sc.pp.normalize_total(expected)
    sc.pp.log1p(expected)
    np.testing.assert_allclose(X_out, np.asarray(expected.X), rtol=1e-9, atol=1e-9)
