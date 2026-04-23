"""Verify the fixture server works."""

import aiohttp
import pytest


@pytest.mark.asyncio
async def test_server_serves_fixture(fixture_server):
    async with aiohttp.ClientSession() as client:
        async with client.get(f"{fixture_server}/pbmc3k.zarr/.zmetadata") as resp:
            assert resp.status == 200
            text = await resp.text()
            assert "zarr_consolidated_format" in text
