from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from cell_explorer_api.services.zarr_adapter import AnnDataZarrAccess


def _make_fake_anndata_store(
    *,
    n_obs: int = 100,
    n_vars: int = 50,
    obsm_keys: list[str] | None = None,
    var_names: list[str] | None = None,
) -> MagicMock:
    store = MagicMock()
    store.n_obs = n_obs
    store.n_vars = n_vars
    store.obsm_keys = list(obsm_keys or ["X_umap", "X_pca"])
    var_df = pd.DataFrame(index=var_names or ["CD8A", "CD4", "MS4A1"])
    store.var = AsyncMock(return_value=var_df)
    return store


@pytest.mark.asyncio
async def test_shape():
    adapter = AnnDataZarrAccess(_make_fake_anndata_store(n_obs=2638, n_vars=1838))
    assert await adapter.shape() == (2638, 1838)


@pytest.mark.asyncio
async def test_attrs_returns_empty_dict():
    adapter = AnnDataZarrAccess(_make_fake_anndata_store())
    assert await adapter.attrs() == {}


@pytest.mark.asyncio
async def test_obsm_keys_copies_list():
    store = _make_fake_anndata_store(obsm_keys=["X_umap", "X_pca"])
    adapter = AnnDataZarrAccess(store)
    keys = await adapter.obsm_keys()
    assert keys == ["X_umap", "X_pca"]
    keys.append("poison")
    # Underlying store must not be mutated
    assert store.obsm_keys == ["X_umap", "X_pca"]


@pytest.mark.asyncio
async def test_var_names_from_var_dataframe():
    store = _make_fake_anndata_store(var_names=["CD8A", "CD4", "MS4A1"])
    adapter = AnnDataZarrAccess(store)
    assert await adapter.var_names() == ["CD8A", "CD4", "MS4A1"]
