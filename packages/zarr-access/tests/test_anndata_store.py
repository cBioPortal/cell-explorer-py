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
