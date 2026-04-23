"""Shared test fixtures for cell2zarr."""
import pytest

from cell2zarr._testing import _create_zarr_store, _write_test_h5ad


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
