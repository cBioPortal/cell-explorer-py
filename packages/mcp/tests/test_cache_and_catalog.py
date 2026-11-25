import asyncio

import httpx
import pytest

from cell_explorer_mcp.catalog import CatalogClient, DatasetNotFoundError
from cell_explorer_mcp.zarr_cache import DatasetCache

from conftest import stub_catalog


class FakeOpener:
    def __init__(self):
        self.opened: list[tuple[str, str]] = []

    async def __call__(self, slug, url, headers=None):
        self.opened.append((slug, url))
        await asyncio.sleep(0.01)
        return type("Entry", (), {"slug": slug, "url": url})()


async def test_cache_reuses_entry():
    opener = FakeOpener()
    cache = DatasetCache(maxsize=2, opener=opener)
    first = await cache.get("a", "u1")
    assert await cache.get("a", "u1") is first
    assert opener.opened == [("a", "u1")]


async def test_cache_concurrent_gets_open_once():
    opener = FakeOpener()
    cache = DatasetCache(maxsize=2, opener=opener)
    results = await asyncio.gather(*(cache.get("a", "u1") for _ in range(5)))
    assert len({id(r) for r in results}) == 1
    assert len(opener.opened) == 1


async def test_cache_evicts_least_recently_used():
    cache = DatasetCache(maxsize=2, opener=FakeOpener())
    await cache.get("a", "u")
    await cache.get("b", "u")
    await cache.get("a", "u")
    await cache.get("c", "u")
    assert "a" in cache and "c" in cache and "b" not in cache
    assert len(cache) == 2


async def test_cache_reopens_when_url_changes():
    opener = FakeOpener()
    cache = DatasetCache(opener=opener)
    await cache.get("a", "old")
    entry = await cache.get("a", "new")
    assert entry.url == "new"
    assert opener.opened == [("a", "old"), ("a", "new")]


async def test_catalog_caches_within_ttl():
    calls: list[str] = []
    catalog = stub_catalog("http://zarr.test", calls)
    await catalog.list_datasets()
    await catalog.get_dataset("pbmc3k")
    await catalog.list_collections()
    assert calls == ["/api/datasets", "/api/collections"]


async def test_catalog_refetches_after_ttl():
    calls: list[str] = []
    catalog = stub_catalog("http://zarr.test", calls, ttl_seconds=0)
    await catalog.list_datasets()
    await catalog.list_datasets()
    assert calls == ["/api/datasets", "/api/datasets"]


async def test_catalog_serves_last_good_list_when_refresh_fails():
    api, calls = {"down": False}, []
    catalog = stub_catalog("http://zarr.test", calls, api, ttl_seconds=0)
    first = await catalog.list_datasets()
    api["down"] = True
    assert await catalog.list_datasets() == first
    assert len(calls) == 2


async def test_catalog_waits_before_retrying_a_failed_refresh():
    api, calls = {"down": False}, []
    catalog = stub_catalog("http://zarr.test", calls, api, ttl_seconds=0)
    await catalog.list_datasets()
    api["down"] = True
    await catalog.list_datasets()
    await catalog.list_datasets()
    assert len(calls) == 2  # the third call is served without retrying


async def test_catalog_retries_after_the_wait():
    api, calls = {"down": False}, []
    catalog = stub_catalog("http://zarr.test", calls, api, ttl_seconds=0, retry_after_seconds=0)
    await catalog.list_datasets()
    api["down"] = True
    await catalog.list_datasets()
    api["down"] = False
    await catalog.list_datasets()
    assert len(calls) == 3


async def test_catalog_fails_without_a_previous_list():
    catalog = stub_catalog("http://zarr.test", api={"down": True})
    with pytest.raises(httpx.HTTPStatusError):
        await catalog.list_datasets()


async def test_catalog_stops_serving_a_list_past_its_max_age():
    api = {"down": False}
    catalog = stub_catalog("http://zarr.test", api=api, ttl_seconds=0, max_stale_seconds=0)
    await catalog.list_datasets()
    api["down"] = True
    with pytest.raises(httpx.HTTPStatusError):
        await catalog.list_datasets()


def test_catalog_default_timeout_is_short():
    assert CatalogClient()._client.timeout.read == 5.0


async def test_catalog_unknown_slug():
    with pytest.raises(DatasetNotFoundError):
        await stub_catalog("http://zarr.test").get_dataset("nope")


async def test_cache_keys_private_entries_by_owner():
    opener = FakeOpener()
    cache = DatasetCache(opener=opener)
    alice = await cache.get("a", "u", headers={"Authorization": "Bearer x"}, owner="alice", ttl=60)
    assert ("a", "bob") not in cache and "a" not in cache
    assert await cache.get("a", "u", owner="bob", ttl=60) is not alice
    assert await cache.get("a", "u", owner="alice", ttl=60) is alice
    assert len(opener.opened) == 2


async def test_catalog_caches_user_listings_apart_for_60s():
    log: list = []
    catalog = stub_catalog("http://zarr.test", log=log)
    assert catalog._user_ttl == 60.0
    mine = await catalog.list_datasets(api_token="api-alice")
    await catalog.list_datasets(api_token="api-alice")
    await catalog.list_datasets(api_token="api-carol")
    shared = await catalog.list_datasets()
    assert [bearer for _, _, bearer in log] == ["api-alice", "api-carol", None]
    assert "secret-cookies" in {d["slug"] for d in mine}
    assert "secret-cookies" not in {d["slug"] for d in shared}
