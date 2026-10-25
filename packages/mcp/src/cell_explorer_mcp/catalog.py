"""Client for the Cell Explorer REST API's dataset and collection listings."""

import asyncio
import time
from typing import Any

import httpx

DEFAULT_API_URL = "https://cell-explorer.cbioportal.org/api"

# Listings change only when an admin adds or edits a dataset.
DEFAULT_TTL_SECONDS = 300.0


class DatasetNotFoundError(LookupError):
    pass


class CatalogClient:
    """Read-only, TTL-cached view of /datasets and /collections."""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        *,
        client: httpx.AsyncClient | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def _get_cached(self, path: str, key: str) -> list[dict[str, Any]]:
        async with self._lock:
            hit = self._cache.get(path)
            if hit is not None and time.monotonic() - hit[0] < self._ttl:
                return hit[1]
            resp = await self._client.get(f"{self._api_url}{path}")
            resp.raise_for_status()
            items = resp.json()[key]
            self._cache[path] = (time.monotonic(), items)
            return items

    async def list_datasets(self) -> list[dict[str, Any]]:
        return await self._get_cached("/datasets", "datasets")

    async def list_collections(self) -> list[dict[str, Any]]:
        return await self._get_cached("/collections", "collections")

    async def get_dataset(self, slug: str) -> dict[str, Any]:
        for dataset in await self.list_datasets():
            if dataset["slug"] == slug:
                return dataset
        raise DatasetNotFoundError(slug)

    async def aclose(self) -> None:
        await self._client.aclose()
