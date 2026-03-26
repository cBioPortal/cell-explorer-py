"""Tests for add-key functionality."""
import numpy as np
import zarr

from cell2zarr.convert import write_obsm_to_store


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
