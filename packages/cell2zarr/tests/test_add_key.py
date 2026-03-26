"""Tests for add-key functionality."""
import numpy as np
import zarr
import anndata as ad
import pandas as pd

from cell2zarr.convert import write_obsm_to_store, add_key_to_store


class TestWriteObsmToStore:
    def test_writes_single_obsm_key(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.open_group(store, mode="w", zarr_format=3)

        obsm_data = {"X_umap": np.random.randn(100, 2).astype(np.float32)}

        write_obsm_to_store(root, obsm_data, n_obs=100, dtype="float32")

        assert "obsm" in root
        assert "X_umap" in root["obsm"]
        arr = root["obsm"]["X_umap"]
        assert arr.shape == (100, 2)

    def test_writes_multiple_obsm_keys(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.open_group(store, mode="w", zarr_format=3)

        obsm_data = {
            "X_umap": np.random.randn(100, 2).astype(np.float32),
            "X_pca": np.random.randn(100, 50).astype(np.float32),
        }

        write_obsm_to_store(root, obsm_data, n_obs=100, dtype="float32")

        assert "X_umap" in root["obsm"]
        assert "X_pca" in root["obsm"]
        assert root["obsm"]["X_pca"].shape == (100, 50)

    def test_respects_dtype(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.open_group(store, mode="w", zarr_format=3)

        obsm_data = {"X_umap": np.random.randn(100, 2).astype(np.float32)}

        write_obsm_to_store(root, obsm_data, n_obs=100, dtype="float16")

        assert root["obsm"]["X_umap"].dtype == np.float16

    def test_empty_obsm_is_noop(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.open_group(store, mode="w", zarr_format=3)

        write_obsm_to_store(root, {}, n_obs=100, dtype="float32")

        assert "obsm" not in root


def _create_test_h5ad(path, n_obs=100, n_vars=50):
    """Create a minimal h5ad with obs, var, obsm, uns."""
    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame({"cell_type": ["A"] * (n_obs // 2) + ["B"] * (n_obs // 2)}, index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame({"gene_name": [f"gene_{i}" for i in range(n_vars)]}, index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = np.random.randn(n_obs, 2).astype(np.float32)
    adata.uns["method"] = "test"
    adata.write_h5ad(path)
    return adata


def _create_test_store(path):
    """Create a minimal empty zarr v3 store."""
    store = zarr.storage.LocalStore(str(path))
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.attrs["encoding-type"] = "anndata"
    root.attrs["encoding-version"] = "0.1.0"
    return root


class TestAddKeyToStore:
    def test_add_obsm(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="obsm", overwrite=False, dtype="float32")

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "obsm" in root
        assert "X_umap" in root["obsm"]
        assert root["obsm"]["X_umap"].shape == (100, 2)

    def test_add_obsm_sub_key(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        adata = _create_test_h5ad(h5ad_path)
        adata.obsm["X_pca"] = np.random.randn(100, 50).astype(np.float32)
        adata.write_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="obsm/X_umap", overwrite=False, dtype="float32")

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "X_umap" in root["obsm"]
        assert "X_pca" not in root["obsm"]

    def test_add_obs(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="obs", overwrite=False, dtype="float32")

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "obs" in root

    def test_add_uns(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="uns", overwrite=False, dtype="float32")

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "uns" in root

    def test_overwrite_fails_when_key_exists(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="obsm", overwrite=False, dtype="float32")

        import pytest
        with pytest.raises(SystemExit):
            add_key_to_store(h5ad_path, zarr_path, key="obsm", overwrite=False, dtype="float32")

    def test_overwrite_succeeds_with_flag(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="obsm", overwrite=False, dtype="float32")
        add_key_to_store(h5ad_path, zarr_path, key="obsm", overwrite=True, dtype="float32")

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "X_umap" in root["obsm"]

    def test_key_not_in_h5ad_fails(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        import pytest
        with pytest.raises(SystemExit):
            add_key_to_store(h5ad_path, zarr_path, key="obsp", overwrite=False, dtype="float32")


class TestAddLargeKeys:
    def test_add_X(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        adata = _create_test_h5ad(h5ad_path, n_obs=100, n_vars=50)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="X", overwrite=False, dtype="float32", temp_dir=tmp_path)

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "X" in root
        assert root["X"].shape == (100, 50)
        np.testing.assert_array_almost_equal(root["X"][:], adata.X, decimal=5)

    def test_add_X_overwrite_fails(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="X", overwrite=False, dtype="float32", temp_dir=tmp_path)

        import pytest
        with pytest.raises(SystemExit):
            add_key_to_store(h5ad_path, zarr_path, key="X", overwrite=False, dtype="float32", temp_dir=tmp_path)

    def test_add_X_overwrite_succeeds(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="X", overwrite=False, dtype="float32", temp_dir=tmp_path)
        add_key_to_store(h5ad_path, zarr_path, key="X", overwrite=True, dtype="float32", temp_dir=tmp_path)

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert root["X"].shape == (100, 50)

    def test_add_layer(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        n_obs, n_vars = 50, 20
        adata = _create_test_h5ad(h5ad_path, n_obs=n_obs, n_vars=n_vars)
        adata.layers["counts"] = np.random.randn(n_obs, n_vars).astype(np.float32)
        adata.write_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="layers/counts", overwrite=False, dtype="float32", temp_dir=tmp_path)

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "layers" in root
        assert "counts" in root["layers"]
        assert root["layers"]["counts"].shape == (n_obs, n_vars)

    def test_add_all_layers(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        n_obs, n_vars = 50, 20
        adata = _create_test_h5ad(h5ad_path, n_obs=n_obs, n_vars=n_vars)
        adata.layers["counts"] = np.random.randn(n_obs, n_vars).astype(np.float32)
        adata.layers["normalized"] = np.random.randn(n_obs, n_vars).astype(np.float32)
        adata.write_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        add_key_to_store(h5ad_path, zarr_path, key="layers", overwrite=False, dtype="float32", temp_dir=tmp_path)

        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "counts" in root["layers"]
        assert "normalized" in root["layers"]

    def test_add_raw_not_supported(self, tmp_path):
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path)
        _create_test_store(zarr_path)

        import pytest
        with pytest.raises(SystemExit):
            add_key_to_store(h5ad_path, zarr_path, key="raw", overwrite=False, dtype="float32")


import subprocess


class TestCLISubcommands:
    def test_convert_default_still_works(self, tmp_path):
        """Invoking without a subcommand should still work (backwards compat)."""
        h5ad_path = tmp_path / "test.h5ad"
        _create_test_h5ad(h5ad_path, n_obs=10, n_vars=5)
        zarr_path = tmp_path / "test.zarr"

        result = subprocess.run(
            ["uv", "run", "cell2zarr", str(h5ad_path), str(zarr_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert zarr_path.exists()

    def test_add_subcommand(self, tmp_path):
        """cell2zarr add should work."""
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"
        _create_test_h5ad(h5ad_path, n_obs=10, n_vars=5)

        # First create a store via convert (use --two-phase to get zarr v3 store)
        convert_result = subprocess.run(
            ["uv", "run", "cell2zarr", str(h5ad_path), str(zarr_path), "--two-phase"],
            capture_output=True, text=True,
        )
        assert convert_result.returncode == 0

        # Add obsm back with --overwrite (tests the add subcommand end-to-end)
        result = subprocess.run(
            ["uv", "run", "cell2zarr", "add", str(h5ad_path), str(zarr_path), "--key", "obsm", "--overwrite"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


class TestEndToEnd:
    def test_convert_then_add_obsm(self, tmp_path):
        """Full workflow: create zarr store, then add obsm with add_key_to_store."""
        h5ad_path = tmp_path / "test.h5ad"
        zarr_path = tmp_path / "test.zarr"

        # Create h5ad with obsm
        n_obs, n_vars = 50, 20
        adata = _create_test_h5ad(h5ad_path, n_obs=n_obs, n_vars=n_vars)
        original_umap = adata.obsm["X_umap"].copy()

        # Create an empty zarr v3 store (simulating convert output)
        _create_test_store(zarr_path)

        # Add obsm using add_key_to_store
        add_key_to_store(h5ad_path, zarr_path, key="obsm", overwrite=False, dtype="float32")

        # Verify obsm is there and correct
        root = zarr.open_group(zarr.storage.LocalStore(str(zarr_path)), mode="r")
        assert "obsm" in root
        assert "X_umap" in root["obsm"]
        assert root["obsm"]["X_umap"].shape == (n_obs, 2)
