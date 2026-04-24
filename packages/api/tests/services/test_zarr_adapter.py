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


import numpy as np


@pytest.mark.asyncio
async def test_obs_column_categorical():
    store = _make_fake_anndata_store(n_obs=5)
    series = pd.Series(
        pd.Categorical(["T cell", "B cell", "T cell", "Monocyte", "B cell"],
                       categories=["T cell", "B cell", "Monocyte"])
    )
    store.obs_column = AsyncMock(return_value=series)

    adapter = AnnDataZarrAccess(store)
    col = await adapter.obs_column("cell_type")

    assert col.dtype == "categorical"
    assert col.categories == ["T cell", "B cell", "Monocyte"]
    np.testing.assert_array_equal(col.values, np.array([0, 1, 0, 2, 1]))


@pytest.mark.asyncio
async def test_obs_column_numeric():
    store = _make_fake_anndata_store(n_obs=3)
    series = pd.Series([1.5, 2.5, 3.5], dtype="float32")
    store.obs_column = AsyncMock(return_value=series)

    adapter = AnnDataZarrAccess(store)
    col = await adapter.obs_column("n_counts")

    assert col.dtype == "numeric"
    assert col.categories is None
    np.testing.assert_allclose(col.values, [1.5, 2.5, 3.5])


@pytest.mark.asyncio
async def test_obs_column_string():
    store = _make_fake_anndata_store(n_obs=3)
    series = pd.Series(["alpha", "beta", "gamma"], dtype="object")
    store.obs_column = AsyncMock(return_value=series)

    adapter = AnnDataZarrAccess(store)
    col = await adapter.obs_column("sample_id")

    assert col.dtype == "string"
    assert col.categories is None
    assert list(col.values) == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_obs_columns_caches_across_calls():
    store = _make_fake_anndata_store()
    # obs_columns on the real AnnDataStore is a property (list[str]).
    store.obs_columns = ["cell_type", "n_counts"]
    # And obs_column returns a Series per column.
    categorical = pd.Series(pd.Categorical(["A", "B"], categories=["A", "B"]))
    numeric = pd.Series([1.0, 2.0], dtype="float32")
    store.obs_column = AsyncMock(side_effect=[categorical, numeric, categorical, numeric])

    adapter = AnnDataZarrAccess(store)
    specs_first = await adapter.obs_columns()
    specs_second = await adapter.obs_columns()

    assert specs_first == specs_second
    # obs_column should have been invoked exactly twice (once per column),
    # not four times: second call hits the cache.
    assert store.obs_column.await_count == 2
    assert [s.name for s in specs_first] == ["cell_type", "n_counts"]
    assert specs_first[0].dtype == "categorical"
    assert specs_first[0].cardinality == 2
    assert specs_first[1].dtype == "numeric"
    assert specs_first[1].cardinality is None


@pytest.mark.asyncio
async def test_gene_index():
    store = _make_fake_anndata_store(var_names=["CD8A", "CD4", "MS4A1"])
    adapter = AnnDataZarrAccess(store)
    assert await adapter.gene_index("CD4") == 1


@pytest.mark.asyncio
async def test_gene_index_unknown_gene_raises_keyerror():
    store = _make_fake_anndata_store(var_names=["CD8A"])
    adapter = AnnDataZarrAccess(store)
    with pytest.raises(KeyError):
        await adapter.gene_index("NOT_A_GENE")


@pytest.mark.asyncio
async def test_gene_column_delegates_to_gene_expression():
    store = _make_fake_anndata_store()
    store.gene_expression = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
    adapter = AnnDataZarrAccess(store)

    values = await adapter.gene_column("CD8A")

    store.gene_expression.assert_awaited_once_with("CD8A")
    np.testing.assert_allclose(values, [0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_obs_mask_categorical():
    store = _make_fake_anndata_store(n_obs=5)
    series = pd.Series(
        pd.Categorical(["T cell", "B cell", "T cell", "Monocyte", "B cell"],
                       categories=["T cell", "B cell", "Monocyte"])
    )
    store.obs_column = AsyncMock(return_value=series)
    adapter = AnnDataZarrAccess(store)

    mask = await adapter.obs_mask("cell_type", "T cell")
    np.testing.assert_array_equal(mask, np.array([True, False, True, False, False]))


@pytest.mark.asyncio
async def test_obs_mask_rejects_numeric():
    store = _make_fake_anndata_store(n_obs=3)
    store.obs_column = AsyncMock(return_value=pd.Series([1.0, 2.0, 3.0]))
    adapter = AnnDataZarrAccess(store)

    with pytest.raises(ValueError, match="categorical"):
        await adapter.obs_mask("n_counts", "1.0")
