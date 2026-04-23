"""Tests for AnnData encoding decoders."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from zarr_access import ZarrStore
from zarr_access.decoders import decode_dataframe, decode_categorical, decode_sparse_matrix


@pytest.mark.asyncio
async def test_decode_dataframe_obs(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    obs_group = await store.get_group("obs")
    df = await decode_dataframe(obs_group)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2638
    assert "louvain" in df.columns
    assert "n_genes" in df.columns


@pytest.mark.asyncio
async def test_decode_categorical(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    louvain_group = await store.get_group("obs/louvain")
    result = await decode_categorical(louvain_group)
    assert isinstance(result, pd.Categorical)
    assert len(result) == 2638


@pytest.mark.asyncio
async def test_decode_sparse_matrix_csr(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    raw_x = await store.get_group("raw/X")
    mat = await decode_sparse_matrix(raw_x)
    assert scipy.sparse.isspmatrix_csr(mat)
    assert mat.shape == (2638, 13714)
