"""Tests for AnnDataStore."""

import numpy as np
import pandas as pd
import pytest

from zarr_access import AnnDataStore, ZarrStore


@pytest.fixture
async def anndata_store(fixture_server):
    zarr_store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    return await AnnDataStore.open(zarr_store)


@pytest.mark.asyncio
async def test_open_validates_encoding_type(fixture_server):
    zarr_store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    store = await AnnDataStore.open(zarr_store)
    assert store.n_obs == 2638
    assert store.n_vars == 1838


@pytest.mark.asyncio
async def test_obsm_keys(anndata_store):
    keys = anndata_store.obsm_keys
    assert "X_umap" in keys
    assert "X_pca" in keys


@pytest.mark.asyncio
async def test_obs_columns(anndata_store):
    cols = anndata_store.obs_columns
    assert "louvain" in cols
    assert "n_genes" in cols


@pytest.mark.asyncio
async def test_obs_dataframe(anndata_store):
    df = await anndata_store.obs()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2638
    assert "louvain" in df.columns


@pytest.mark.asyncio
async def test_var_dataframe(anndata_store):
    df = await anndata_store.var()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1838


@pytest.mark.asyncio
async def test_obs_column(anndata_store):
    result = await anndata_store.obs_column("n_genes")
    assert isinstance(result, np.ndarray)
    assert len(result) == 2638


@pytest.mark.asyncio
async def test_obsm_umap(anndata_store):
    umap = await anndata_store.obsm("X_umap")
    assert isinstance(umap, np.ndarray)
    assert umap.shape == (2638, 2)


@pytest.mark.asyncio
async def test_gene_expression(anndata_store):
    var_df = await anndata_store.var()
    gene = var_df.index[0]
    expr = await anndata_store.gene_expression(gene)
    assert isinstance(expr, np.ndarray)
    assert len(expr) == 2638


@pytest.mark.asyncio
async def test_X_dense(anndata_store):
    """pbmc3k.zarr has dense X."""
    x = await anndata_store.X()
    assert x.shape == (2638, 1838)
