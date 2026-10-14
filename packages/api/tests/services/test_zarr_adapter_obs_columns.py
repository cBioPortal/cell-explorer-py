"""AnnDataZarrAccess.obs_columns() describes columns from metadata, not cell data."""

import pytest
from zarr_access import AnnDataStore, ZarrStore

from cell_explorer_api.services.zarr_adapter import AnnDataZarrAccess

# zarr_server runs on the session loop; see its docstring in conftest.py.
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _open(base_url: str, name: str) -> tuple[ZarrStore, AnnDataStore]:
    zs = await ZarrStore.open(f"{base_url}/{name}")
    return zs, await AnnDataStore.open(zs)


def _by_name(specs):
    return {s.name: (s.dtype, s.cardinality, s.categories) for s in specs}


async def _full_read(anndata: AnnDataStore):
    """The pre-metadata behaviour: decode every column in full."""
    return _by_name(await AnnDataZarrAccess(anndata).obs_columns())


@pytest.mark.parametrize("store", ["chat_obs_v3.zarr", "chat_obs_v2.zarr"])
async def test_matches_full_read_with_names_capped_at_50(zarr_server, store):
    zs, anndata = await _open(zarr_server, store)
    fast = _by_name(await AnnDataZarrAccess(anndata, zarr_store=zs).obs_columns())

    expected = {
        name: (dtype, card, cats if card is None or card <= 50 else None)
        for name, (dtype, card, cats) in (await _full_read(anndata)).items()
    }
    assert fast == expected


async def test_describes_each_column_kind(zarr_server):
    zs, anndata = await _open(zarr_server, "chat_obs_v3.zarr")
    cols = _by_name(await AnnDataZarrAccess(anndata, zarr_store=zs).obs_columns())

    assert cols["cell_type"] == ("categorical", 3, ["B", "NK", "T"])
    assert cols["donor"] == ("categorical", 60, None)
    assert cols["n_counts"] == ("numeric", None, None)
    assert cols["n_genes"] == ("numeric", None, None)
    # Booleans stay "string", as the full read has always reported them.
    assert cols["is_primary"] == ("string", None, None)
    assert cols["barcode"] == ("string", None, None)


async def test_reads_no_per_cell_data(zarr_server, monkeypatch):
    zs, anndata = await _open(zarr_server, "chat_obs_v3.zarr")

    async def no_column_reads(name):
        raise AssertionError(f"read whole column {name!r}")

    monkeypatch.setattr(anndata, "obs_column", no_column_reads)
    read_paths: list[str] = []
    real_get_array = zs.get_array

    async def spy(path):
        read_paths.append(path)
        return await real_get_array(path)

    monkeypatch.setattr(zs, "get_array", spy)

    await AnnDataZarrAccess(anndata, zarr_store=zs).obs_columns()
    # Only the categories of columns under the cap; never codes or values.
    assert read_paths == ["obs/cell_type/categories"]


async def test_falls_back_to_full_read_without_consolidated_metadata(zarr_server):
    zs, anndata = await _open(zarr_server, "chat_obs_v2_unconsolidated.zarr")
    assert not zs.consolidated_metadata

    cols = _by_name(await AnnDataZarrAccess(anndata, zarr_store=zs).obs_columns())
    assert cols == await _full_read(anndata)
