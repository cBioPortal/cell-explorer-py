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


@pytest.mark.asyncio
async def test_open_with_auth_header(auth_fixture_server):
    base_url, token = auth_fixture_server
    store = await ZarrStore.open(
        f"{base_url}/pbmc3k.zarr",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert store.zarr_version == 2


@pytest.mark.asyncio
async def test_open_without_auth_fails(auth_fixture_server):
    base_url, _ = auth_fixture_server
    with pytest.raises(Exception):
        await ZarrStore.open(f"{base_url}/pbmc3k.zarr")


@pytest.mark.asyncio
async def test_open_sends_default_origin(header_capture_server, monkeypatch):
    """ZarrStore must send a deterministic Origin header on every request to
    avoid CDN cache poisoning (CloudFront caches no-Origin S3 responses without
    CORS headers and serves them to browsers that DO send Origin)."""
    monkeypatch.delenv("ZARR_ACCESS_ORIGIN", raising=False)
    base_url, recorded = header_capture_server
    await ZarrStore.open(f"{base_url}/pbmc3k.zarr")
    assert recorded, "expected at least one HTTP request"
    assert all(o == "https://cell-explorer.cbioportal.org" for o in recorded), recorded


@pytest.mark.asyncio
async def test_open_origin_overridable_via_env(header_capture_server, monkeypatch):
    monkeypatch.setenv("ZARR_ACCESS_ORIGIN", "https://my-deployment.example")
    base_url, recorded = header_capture_server
    await ZarrStore.open(f"{base_url}/pbmc3k.zarr")
    assert all(o == "https://my-deployment.example" for o in recorded), recorded


@pytest.mark.asyncio
async def test_open_caller_origin_takes_precedence(header_capture_server):
    """If the caller passes an Origin in headers, it wins over the default."""
    base_url, recorded = header_capture_server
    await ZarrStore.open(
        f"{base_url}/pbmc3k.zarr",
        headers={"Origin": "https://caller-supplied.example"},
    )
    assert all(o == "https://caller-supplied.example" for o in recorded), recorded
