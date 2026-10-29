import asyncio

import pytest

from cell_explorer_mcp.catalog import DatasetNotFoundError
from cell_explorer_mcp.zarr_cache import DatasetCache

from conftest import stub_catalog


class FakeOpener:
    def __init__(self):
        self.opened: list[tuple[str, str]] = []

    async def __call__(self, slug, url):
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


async def test_catalog_unknown_slug():
    with pytest.raises(DatasetNotFoundError):
        await stub_catalog("http://zarr.test").get_dataset("nope")
