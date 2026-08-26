"""Tests for the zarr store metadata extractor."""

import pytest
from zarr_access import ZarrStore

from cell_explorer_api.services.store_metadata import extract_store_metadata

# zarr_server is a session-scoped async fixture (it starts one aiohttp server
# for the whole test session). The root pyproject.toml defaults async test
# loops to function scope, so without this marker each test runs on its own
# event loop while the server keeps running on the session-scoped loop from
# fixture setup — the server's loop never gets to iterate during the test,
# so client requests hang forever. Pinning these tests to loop_scope="session"
# puts them on the same loop as the fixture.
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _extract(base_url: str, name: str):
    store = await ZarrStore.open(f"{base_url}/{name}")
    return await extract_store_metadata(store)


async def test_extracts_shape_from_v3_store(zarr_server):
    md = await _extract(zarr_server, "tiny_v3.zarr")
    assert md.n_obs == 12
    assert md.n_vars == 5
    assert md.zarr_version == 3


async def test_extracts_shape_from_v2_store(zarr_server):
    md = await _extract(zarr_server, "tiny_v2.zarr")
    assert md.n_obs == 12
    assert md.n_vars == 5
    assert md.zarr_version == 2


async def test_extracts_obsm_and_columns(zarr_server):
    md = await _extract(zarr_server, "tiny_v3.zarr")
    assert md.obsm_keys == ["X_umap"]
    assert set(md.obs_columns) == {"cell_type", "n_counts"}
    assert md.var_columns == ["feature_name"]


async def test_extracts_layers_and_x_encoding(zarr_server):
    md = await _extract(zarr_server, "tiny_v3.zarr")
    assert md.layers == ["counts"]
    assert md.x_encoding == "array"
    assert md.x_dtype == "float32"


async def test_non_anndata_store_raises(zarr_server):
    with pytest.raises(Exception):
        await _extract(zarr_server, "not_anndata.zarr")


async def test_anndata_without_x_raises(zarr_server):
    with pytest.raises(Exception):
        await _extract(zarr_server, "no_x.zarr")
