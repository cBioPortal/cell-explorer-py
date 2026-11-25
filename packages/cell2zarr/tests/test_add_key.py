"""Tests for add-key functionality."""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.io import write_elem
from click.testing import CliRunner
from scipy import sparse

from cell2zarr.cli import cli
from cell2zarr.convert import write_obsm_to_store, add_key_to_store
from cell2zarr._testing import _write_test_h5ad, _create_zarr_store, open_zarr


class TestWriteObsmToStore:
    @pytest.fixture
    def zarr_root(self, tmp_path):
        store = zarr.storage.LocalStore(str(tmp_path / "test.zarr"))
        return zarr.open_group(store, mode="w", zarr_format=3)

    def test_writes_single_key(self, zarr_root):
        obsm_data = {"X_umap": np.random.randn(100, 2).astype(np.float32)}
        write_obsm_to_store(zarr_root, obsm_data, n_obs=100, dtype="float32")

        assert "obsm" in zarr_root
        assert zarr_root["obsm"]["X_umap"].shape == (100, 2)

    def test_writes_multiple_keys(self, zarr_root):
        obsm_data = {
            "X_umap": np.random.randn(100, 2).astype(np.float32),
            "X_pca": np.random.randn(100, 50).astype(np.float32),
        }
        write_obsm_to_store(zarr_root, obsm_data, n_obs=100, dtype="float32")

        assert "X_umap" in zarr_root["obsm"]
        assert zarr_root["obsm"]["X_pca"].shape == (100, 50)

    @pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
    def test_respects_dtype(self, zarr_root, dtype):
        obsm_data = {"X_umap": np.random.randn(100, 2).astype(np.float32)}
        write_obsm_to_store(zarr_root, obsm_data, n_obs=100, dtype=dtype)

        assert zarr_root["obsm"]["X_umap"].dtype == np.dtype(dtype)

    def test_empty_obsm_is_noop(self, zarr_root):
        write_obsm_to_store(zarr_root, {}, n_obs=100, dtype="float32")
        assert "obsm" not in zarr_root


class TestAddSmallKeys:
    @pytest.mark.parametrize("key", ["obsm", "obs", "uns"])
    def test_add_key(self, sample_h5ad, sample_store, key):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(h5ad_path, sample_store, key=key, overwrite=False, dtype="float32")

        root = open_zarr(sample_store)
        assert key in root

    def test_add_obsm_has_correct_shape(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(h5ad_path, sample_store, key="obsm", overwrite=False, dtype="float32")

        root = open_zarr(sample_store)
        assert root["obsm"]["X_umap"].shape == (100, 2)

    def test_add_obsm_sub_key(self, h5ad_path, sample_store):
        path, adata = _write_test_h5ad(h5ad_path)
        adata.obsm["X_pca"] = np.random.randn(100, 50).astype(np.float32)
        adata.write_h5ad(path)

        add_key_to_store(path, sample_store, key="obsm/X_umap", overwrite=False, dtype="float32")

        root = open_zarr(sample_store)
        assert "X_umap" in root["obsm"]
        assert "X_pca" not in root["obsm"]

    def test_key_not_in_h5ad_fails(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        with pytest.raises(SystemExit):
            add_key_to_store(h5ad_path, sample_store, key="obsp", overwrite=False, dtype="float32")


class TestOverwrite:
    def test_fails_without_flag(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(h5ad_path, sample_store, key="obsm", overwrite=False, dtype="float32")

        with pytest.raises(SystemExit):
            add_key_to_store(h5ad_path, sample_store, key="obsm", overwrite=False, dtype="float32")

    def test_succeeds_with_flag(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(h5ad_path, sample_store, key="obsm", overwrite=False, dtype="float32")
        add_key_to_store(h5ad_path, sample_store, key="obsm", overwrite=True, dtype="float32")

        root = open_zarr(sample_store)
        assert "X_umap" in root["obsm"]

    @pytest.mark.parametrize("key", ["X", "obsm"])
    def test_overwrite_pattern(self, sample_h5ad, sample_store, tmp_path, key):
        """Both small and large keys follow same overwrite pattern."""
        h5ad_path, _ = sample_h5ad
        add_key_to_store(h5ad_path, sample_store, key=key, overwrite=False, dtype="float32", temp_dir=tmp_path)

        with pytest.raises(SystemExit):
            add_key_to_store(h5ad_path, sample_store, key=key, overwrite=False, dtype="float32", temp_dir=tmp_path)

        add_key_to_store(h5ad_path, sample_store, key=key, overwrite=True, dtype="float32", temp_dir=tmp_path)


class TestAddLargeKeys:
    def test_add_X(self, sample_h5ad, sample_store, tmp_path):
        h5ad_path, adata = sample_h5ad
        add_key_to_store(h5ad_path, sample_store, key="X", overwrite=False, dtype="float32", temp_dir=tmp_path)

        root = open_zarr(sample_store)
        assert root["X"].shape == (100, 50)
        np.testing.assert_array_almost_equal(root["X"][:], adata.X, decimal=5)

    def test_add_single_layer(self, h5ad_path, sample_store, tmp_path):
        path, adata = _write_test_h5ad(h5ad_path, n_obs=50, n_vars=20)
        adata.layers["counts"] = np.random.randn(50, 20).astype(np.float32)
        adata.write_h5ad(path)

        add_key_to_store(path, sample_store, key="layers/counts", overwrite=False, dtype="float32", temp_dir=tmp_path)

        root = open_zarr(sample_store)
        assert root["layers"]["counts"].shape == (50, 20)

    def test_add_all_layers(self, h5ad_path, sample_store, tmp_path):
        path, adata = _write_test_h5ad(h5ad_path, n_obs=50, n_vars=20)
        adata.layers["counts"] = np.random.randn(50, 20).astype(np.float32)
        adata.layers["normalized"] = np.random.randn(50, 20).astype(np.float32)
        adata.write_h5ad(path)

        add_key_to_store(path, sample_store, key="layers", overwrite=False, dtype="float32", temp_dir=tmp_path)

        root = open_zarr(sample_store)
        assert "counts" in root["layers"]
        assert "normalized" in root["layers"]



def _write_h5ad_with_raw(path, n_obs=40, n_vars=10, n_raw_vars=25, varm_key="PCs"):
    """h5ad whose raw keeps more genes than X, as after gene filtering. Returns (path, adata)."""
    rng = np.random.default_rng(0)
    counts = sparse.random(n_obs, n_raw_vars, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    full = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(
            {"gene_symbol": [f"SYM{i}" for i in range(n_raw_vars)]},
            index=[f"gene_{i}" for i in range(n_raw_vars)],
        ),
    )
    full.varm[varm_key] = rng.standard_normal((n_raw_vars, 3)).astype(np.float32)
    full.raw = full
    adata = full[:, :n_vars].copy()
    adata.write_h5ad(path)
    return path, adata


class TestAddRaw:
    def test_add_raw_writes_x_var_and_varm(self, h5ad_path, sample_store, tmp_path):
        path, adata = _write_h5ad_with_raw(h5ad_path)
        add_key_to_store(path, sample_store, key="raw", overwrite=False, dtype="float32", temp_dir=tmp_path)

        raw = open_zarr(sample_store)["raw"]
        assert raw.attrs["encoding-type"] == "raw"
        # raw.X keeps raw's own gene count, not X's.
        assert raw["X"].shape == (40, 25)
        np.testing.assert_array_almost_equal(raw["X"][:], adata.raw.X.toarray(), decimal=5)
        assert raw["var"]["gene_symbol"].attrs["encoding-type"] == "categorical"
        assert raw["varm"]["PCs"].shape == (25, 3)

    def test_anndata_reads_added_raw(self, h5ad_path, sample_store, tmp_path):
        path, adata = _write_h5ad_with_raw(h5ad_path)
        for key in ("obs", "var", "X", "raw"):
            add_key_to_store(path, sample_store, key=key, overwrite=False, dtype="float32", temp_dir=tmp_path)

        back = ad.read_zarr(sample_store)
        assert back.shape == (40, 10)
        assert back.raw.shape == (40, 25)
        np.testing.assert_array_almost_equal(np.asarray(back.raw.X), adata.raw.X.toarray(), decimal=5)
        assert list(back.raw.var_names) == list(adata.raw.var_names)
        assert list(back.raw.var["gene_symbol"]) == list(adata.raw.var["gene_symbol"])
        np.testing.assert_array_almost_equal(back.raw.varm["PCs"], adata.raw.varm["PCs"])

    def test_respects_dtype(self, h5ad_path, sample_store, tmp_path):
        path, _ = _write_h5ad_with_raw(h5ad_path)
        add_key_to_store(path, sample_store, key="raw", overwrite=False, dtype="float16", temp_dir=tmp_path)

        assert open_zarr(sample_store)["raw"]["X"].dtype == np.float16

    def test_raw_not_in_h5ad_fails(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        with pytest.raises(SystemExit):
            add_key_to_store(h5ad_path, sample_store, key="raw", overwrite=False, dtype="float32")

    def test_sub_key_rejected(self, h5ad_path, sample_store):
        path, _ = _write_h5ad_with_raw(h5ad_path)
        with pytest.raises(SystemExit):
            add_key_to_store(path, sample_store, key="raw/X", overwrite=False, dtype="float32")

    def test_fails_without_overwrite(self, h5ad_path, sample_store, tmp_path):
        path, _ = _write_h5ad_with_raw(h5ad_path)
        add_key_to_store(path, sample_store, key="raw", overwrite=False, dtype="float32", temp_dir=tmp_path)

        with pytest.raises(SystemExit):
            add_key_to_store(path, sample_store, key="raw", overwrite=False, dtype="float32", temp_dir=tmp_path)

    def test_overwrite_replaces_every_element(self, tmp_path, sample_store):
        first, _ = _write_h5ad_with_raw(tmp_path / "first.h5ad", n_raw_vars=25, varm_key="PCs")
        second, adata = _write_h5ad_with_raw(tmp_path / "second.h5ad", n_raw_vars=30, varm_key="loadings")
        add_key_to_store(first, sample_store, key="raw", overwrite=False, dtype="float32", temp_dir=tmp_path)
        add_key_to_store(second, sample_store, key="raw", overwrite=True, dtype="float32", temp_dir=tmp_path)

        raw = open_zarr(sample_store)["raw"]
        assert raw["X"].shape == (40, 30)
        np.testing.assert_array_almost_equal(raw["X"][:], adata.raw.X.toarray(), decimal=5)
        assert list(raw["varm"].keys()) == ["loadings"]

    def test_overwrite_replaces_anndata_written_raw(self, h5ad_path, sample_store, tmp_path):
        """A raw anndata wrote itself (sparse X, a group) is replaced by a dense one."""
        path, adata = _write_h5ad_with_raw(h5ad_path)
        root = zarr.open_group(zarr.storage.LocalStore(str(sample_store)), mode="r+")
        write_elem(root, "raw", adata.raw)
        assert root["raw"]["X"].attrs["encoding-type"] == "csr_matrix"

        add_key_to_store(path, sample_store, key="raw", overwrite=True, dtype="float32", temp_dir=tmp_path)

        raw = open_zarr(sample_store)["raw"]
        assert isinstance(raw["X"], zarr.Array)
        assert raw["X"].attrs["encoding-type"] == "array"
        np.testing.assert_array_almost_equal(raw["X"][:], adata.raw.X.toarray(), decimal=5)


class TestCLI:
    def test_convert_default(self, sample_h5ad, zarr_path):
        h5ad_path, _ = sample_h5ad
        result = CliRunner().invoke(cli, [str(h5ad_path), str(zarr_path)])
        assert result.exit_code == 0, result.output
        assert zarr_path.exists()

    def test_convert_explicit(self, sample_h5ad, zarr_path):
        h5ad_path, _ = sample_h5ad
        result = CliRunner().invoke(cli, ["convert", str(h5ad_path), str(zarr_path)])
        assert result.exit_code == 0, result.output
        assert zarr_path.exists()

    def test_add_subcommand(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        result = CliRunner().invoke(cli, ["add", str(h5ad_path), str(sample_store), "--key", "obsm"])
        assert result.exit_code == 0, result.output

    def test_add_raw_subcommand(self, h5ad_path, sample_store, tmp_path):
        path, _ = _write_h5ad_with_raw(h5ad_path)
        result = CliRunner().invoke(
            cli, ["add", str(path), str(sample_store), "--key", "raw", "--temp-dir", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        assert open_zarr(sample_store)["raw"]["X"].shape == (40, 25)


class TestEndToEnd:
    def test_convert_then_add_obsm(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(h5ad_path, sample_store, key="obsm", overwrite=False, dtype="float32")

        root = open_zarr(sample_store)
        assert "obsm" in root
        assert root["obsm"]["X_umap"].shape == (100, 2)


def _root_metadata(store_path):
    """Read the raw root zarr.json of a store."""
    return json.loads((store_path / "zarr.json").read_text())


class TestAddKeyConsolidation:
    def test_consolidates_by_default(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(h5ad_path, sample_store, key="obs", overwrite=False, dtype="float32")

        meta = _root_metadata(sample_store)
        assert "consolidated_metadata" in meta
        assert "obs" in meta["consolidated_metadata"]["metadata"]

    def test_no_consolidate_leaves_root_metadata_untouched(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(
            h5ad_path, sample_store, key="obs", overwrite=False, dtype="float32",
            consolidate=False,
        )

        meta = _root_metadata(sample_store)
        assert "consolidated_metadata" not in meta

    def test_no_consolidate_still_writes_the_group(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(
            h5ad_path, sample_store, key="obs", overwrite=False, dtype="float32",
            consolidate=False,
        )

        assert (sample_store / "obs").is_dir()
        assert (sample_store / "obs" / "zarr.json").exists()

    def test_cli_no_consolidate_flag(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["add", str(h5ad_path), str(sample_store), "--key", "obs", "--no-consolidate"],
        )

        assert result.exit_code == 0, result.output
        assert "consolidated_metadata" not in _root_metadata(sample_store)

    def test_cli_consolidates_without_the_flag(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        runner = CliRunner()
        result = runner.invoke(
            cli, ["add", str(h5ad_path), str(sample_store), "--key", "obs"]
        )

        assert result.exit_code == 0, result.output
        assert "consolidated_metadata" in _root_metadata(sample_store)


class TestConsolidateStore:
    def test_consolidates_after_unconsolidated_adds(self, sample_h5ad, sample_store):
        from cell2zarr.convert import consolidate_store

        h5ad_path, _ = sample_h5ad
        for key in ("obs", "obsm", "uns"):
            add_key_to_store(
                h5ad_path, sample_store, key=key, overwrite=False,
                dtype="float32", consolidate=False,
            )
        assert "consolidated_metadata" not in _root_metadata(sample_store)

        consolidate_store(sample_store)

        listed = _root_metadata(sample_store)["consolidated_metadata"]["metadata"]
        assert "obs" in listed
        assert "obsm" in listed
        assert "uns" in listed

    def test_missing_store_exits_nonzero(self, tmp_path):
        from cell2zarr.convert import consolidate_store

        with pytest.raises(SystemExit) as excinfo:
            consolidate_store(tmp_path / "does_not_exist.zarr")
        assert excinfo.value.code == 1

    def test_cli_consolidate(self, sample_h5ad, sample_store):
        h5ad_path, _ = sample_h5ad
        add_key_to_store(
            h5ad_path, sample_store, key="obs", overwrite=False,
            dtype="float32", consolidate=False,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["consolidate", str(sample_store)])

        assert result.exit_code == 0, result.output
        assert "obs" in _root_metadata(sample_store)["consolidated_metadata"]["metadata"]
