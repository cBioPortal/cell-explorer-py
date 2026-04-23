"""Tests for ZarrStore."""

import pytest

from zarr_access import ZarrStore


@pytest.mark.asyncio
async def test_open_v2(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    assert store.zarr_version == 2


@pytest.mark.asyncio
async def test_open_v3(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k_v3.zarr")
    assert store.zarr_version == 3


@pytest.mark.asyncio
async def test_v2_has_consolidated_metadata(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    assert store.consolidated_metadata is not None
    assert "X/.zarray" in store.consolidated_metadata


@pytest.mark.asyncio
async def test_get_array_v2(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    arr = await store.get_array("X")
    assert arr.shape == (2638, 1838)


@pytest.mark.asyncio
async def test_get_group_v2(fixture_server):
    store = await ZarrStore.open(f"{fixture_server}/pbmc3k.zarr")
    group = await store.get_group("obs")
    assert group is not None
