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
