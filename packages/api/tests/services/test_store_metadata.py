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
    assert set(md.obs_columns) == {
        "cell_type", "tissue", "tissue_ontology_term_id",
        "observation_joinid", "n_counts",
    }
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


async def test_reads_values_for_a_categorical_column_under_the_cap(zarr_server):
    md = await _extract(zarr_server, "tiny_v3.zarr")
    tissue = md.obs_facets["tissue"]
    assert tissue.dtype == "categorical"
    assert tissue.cardinality == 3
    assert sorted(tissue.values) == ["brain", "liver", "lung"]


async def test_reads_values_from_a_v2_store_too(zarr_server):
    md = await _extract(zarr_server, "tiny_v2.zarr")
    assert sorted(md.obs_facets["tissue"].values) == ["brain", "liver", "lung"]


async def test_reports_cardinality_but_no_values_over_the_cap(zarr_server, monkeypatch):
    from cell_explorer_api.services import store_metadata as sm

    calls: list[str] = []
    real = sm._read_categories

    async def spy(store, column):
        calls.append(column)
        return await real(store, column)

    monkeypatch.setattr(sm, "_read_categories", spy)
    monkeypatch.setattr(sm, "FACET_VALUE_CAP", 2)
    md = await _extract(zarr_server, "tiny_v3.zarr")

    assert md.obs_facets["tissue"].cardinality == 3, "cardinality is free from metadata, always reported"
    assert md.obs_facets["tissue"].values is None, "over the cap, values must not be read"
    assert "tissue" not in calls, "an over-cap column must never be read at all"
    assert "cell_type" in calls, "the spy is wired up and under-cap columns still read"


async def test_excludes_ontology_term_id_columns(zarr_server):
    md = await _extract(zarr_server, "tiny_v3.zarr")
    twin = md.obs_facets["tissue_ontology_term_id"]
    assert twin.cardinality == 1
    assert twin.values is None, "ontology ids are never shipped as facet values"


async def test_records_numeric_columns_without_values(zarr_server):
    md = await _extract(zarr_server, "tiny_v3.zarr")
    numeric = md.obs_facets["n_counts"]
    assert numeric.dtype == "numeric"
    assert numeric.cardinality is None
    assert numeric.values is None


async def test_records_numeric_columns_in_a_v2_store_too(zarr_server):
    # v2's dtype lives on .zarray, not .zattrs — a regression here silently
    # reports every v2 numeric column as "string" instead of "numeric".
    md = await _extract(zarr_server, "tiny_v2.zarr")
    numeric = md.obs_facets["n_counts"]
    assert numeric.dtype == "numeric"
    assert numeric.cardinality is None
    assert numeric.values is None


async def test_no_consolidated_metadata_yields_empty_obs_facets(zarr_server):
    # A store lacking both zarr.json consolidated metadata and .zmetadata has
    # consolidated_metadata == None. Fabricating dtype "string" for every
    # column here would be a false positive claim ("looked, it's a string")
    # rather than the true state ("never looked"). The harvest must still
    # succeed on counts even though obs_facets comes back empty.
    store = await ZarrStore.open(f"{zarr_server}/tiny_v3.zarr")
    store._consolidated_metadata = None
    md = await extract_store_metadata(store)
    assert md.obs_facets == {}
    assert md.n_obs == 12
    assert md.n_vars == 5


async def test_a_failing_values_read_does_not_fail_the_harvest(zarr_server, monkeypatch):
    # One unreadable column must cost that column, not the dataset.
    from cell_explorer_api.services import store_metadata as sm

    real = sm._read_categories

    async def flaky(store, column):
        if column == "tissue":
            raise OSError("chunk unreadable")
        return await real(store, column)

    monkeypatch.setattr(sm, "_read_categories", flaky)
    md = await _extract(zarr_server, "tiny_v3.zarr")
    assert md.obs_facets["tissue"].values is None
    assert md.obs_facets["cell_type"].values is not None, "other columns still read"
    assert md.n_obs == 12, "counts unaffected"
