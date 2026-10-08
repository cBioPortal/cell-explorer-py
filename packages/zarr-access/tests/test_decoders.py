"""Tests for AnnData encoding decoders."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from zarr_access import ZarrStore
from zarr_access.decoders import decode_dataframe, decode_categorical, decode_sparse_matrix, decode_column, decode_nullable_string


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


@pytest.mark.asyncio
async def test_decode_dataframe_reads_nullable_string_index(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/nullable_strings.zarr")
    obs = await decode_dataframe(await store.get_group("obs"))
    assert obs.index.tolist() == [f"cell_{i}" for i in range(6)]
    var = await decode_dataframe(await store.get_group("var"))
    assert var.index.tolist() == ["gene_0", "gene_1", "gene_2"]


@pytest.mark.asyncio
async def test_decode_nullable_string_masks_missing_values(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/nullable_strings.zarr")
    values = await decode_nullable_string(await store.get_group("obs/donor"))
    assert values.tolist() == ["d1", None, "d2", "d1", None, "d2"]


@pytest.mark.asyncio
async def test_decode_column_handles_nullable_string_group(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/nullable_strings.zarr")
    values = await decode_column(await store.get_group("obs/donor"))
    assert values.tolist() == ["d1", None, "d2", "d1", None, "d2"]


@pytest.mark.asyncio
async def test_decode_dataframe_keeps_other_columns_beside_nullable_ones(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/nullable_strings.zarr")
    obs = await decode_dataframe(await store.get_group("obs"))
    assert obs["cell_type"].tolist() == ["T", "B", "T", "B", "T", "B"]
    assert obs["donor"].tolist() == ["d1", None, "d2", "d1", None, "d2"]


@pytest.mark.asyncio
async def test_decode_dataframe_keeps_column_dtypes_for_old_stores(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    df = await decode_dataframe(await store.get_group("obs"))
    assert isinstance(df["louvain"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_numeric_dtype(df["n_genes"])
    assert pd.api.types.is_numeric_dtype(df["percent_mito"])
