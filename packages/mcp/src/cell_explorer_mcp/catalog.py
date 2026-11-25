"""Client for the Cell Explorer REST API's dataset and collection listings."""

import asyncio
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://cell-explorer.cbioportal.org/api"

# Listings change only when an admin adds or edits a dataset.
DEFAULT_TTL_SECONDS = 300.0
# A healthy fetch takes 0.2-0.4s; while it runs every other lookup waits.
DEFAULT_TIMEOUT_SECONDS = 5.0
# When a refresh fails, the last good listing keeps being served (up to
# MAX_STALE) and the API is retried at most once per RETRY_AFTER.
DEFAULT_RETRY_AFTER_SECONDS = 30.0
DEFAULT_MAX_STALE_SECONDS = 3600.0


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
        retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
        max_stale_seconds: float = DEFAULT_MAX_STALE_SECONDS,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        self._ttl = ttl_seconds
        self._retry_after = retry_after_seconds
        self._max_stale = max_stale_seconds
        self._cache: dict[str, tuple[float, Any]] = {}
        self._failed_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def _get_cached(self, path: str, key: str) -> list[dict[str, Any]]:
        async with self._lock:
            hit = self._cache.get(path)
            now = time.monotonic()
            usable = hit is not None and now - hit[0] < self._max_stale
            if hit is not None and now - hit[0] < self._ttl:
                return hit[1]
            if usable and now - self._failed_at.get(path, float("-inf")) < self._retry_after:
                return hit[1]
            try:
                resp = await self._client.get(f"{self._api_url}{path}")
                resp.raise_for_status()
                items = resp.json()[key]
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                if not usable:
                    raise
                self._failed_at[path] = time.monotonic()
                logger.warning("catalog refresh of %s failed (%s); serving the last good listing", path, exc)
                return hit[1]
            self._cache[path] = (time.monotonic(), items)
            self._failed_at.pop(path, None)
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
