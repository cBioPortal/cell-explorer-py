"""Client for the Cell Explorer REST API's dataset and collection listings."""

import asyncio
import hashlib
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
# A signed-in user's listings: short-lived, so a revoked role shows soon.
DEFAULT_USER_TTL_SECONDS = 60.0
MAX_USER_CACHE_ENTRIES = 2000


class DatasetNotFoundError(LookupError):
    pass


class DatasetAccessDeniedError(PermissionError):
    pass


# The cookies a signed-cookie credential consists of.
CLOUDFRONT_COOKIES = ("CloudFront-Policy", "CloudFront-Signature", "CloudFront-Key-Pair-Id")


def credential_headers(credential: dict[str, Any]) -> dict[str, str]:
    """HTTP headers that open a store with an /access credential."""
    kind = credential.get("credential_type")
    if kind == "public":
        return {}
    if kind == "bearer_token":
        return {"Authorization": f"Bearer {credential['token']}"}
    if kind == "signed_cookies":
        return {"Cookie": "; ".join(f"{k}={v}" for k, v in credential["cookies"].items())}
    raise ValueError(f"unknown credential_type {kind!r}")


def _cloudfront_cookies(resp: httpx.Response) -> dict[str, str]:
    """The CloudFront cookies among the response's Set-Cookie headers."""
    cookies = {}
    for header in resp.headers.get_list("set-cookie"):
        name, _, value = header.split(";", 1)[0].strip().partition("=")
        if name in CLOUDFRONT_COOKIES:
            cookies[name] = value
    return cookies


class CatalogClient:
    """Read-only, TTL-cached view of /datasets and /collections.

    Anonymous listings share one cache with a stale fallback. A signed-in
    user's listings (api_token given) are cached apart, per token, and never
    merged into the shared one.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        *,
        client: httpx.AsyncClient | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
        max_stale_seconds: float = DEFAULT_MAX_STALE_SECONDS,
        user_ttl_seconds: float = DEFAULT_USER_TTL_SECONDS,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        self._ttl = ttl_seconds
        self._retry_after = retry_after_seconds
        self._max_stale = max_stale_seconds
        self._cache: dict[str, tuple[float, Any]] = {}
        self._failed_at: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._user_ttl = user_ttl_seconds
        self._user_cache: dict[tuple[str, str], tuple[float, Any]] = {}

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

    async def _get_for_user(self, path: str, key: str, api_token: str) -> list[dict[str, Any]]:
        cache_key = (hashlib.sha256(api_token.encode()).hexdigest(), path)
        hit = self._user_cache.get(cache_key)
        if hit is not None and time.monotonic() - hit[0] < self._user_ttl:
            return hit[1]
        resp = await self._client.get(
            f"{self._api_url}{path}", headers={"Authorization": f"Bearer {api_token}"}
        )
        resp.raise_for_status()
        items = resp.json()[key]
        now = time.monotonic()
        if cache_key not in self._user_cache and len(self._user_cache) >= MAX_USER_CACHE_ENTRIES:
            self._user_cache = {
                k: v for k, v in self._user_cache.items() if now - v[0] < self._user_ttl
            }
            if len(self._user_cache) >= MAX_USER_CACHE_ENTRIES:
                self._user_cache.pop(next(iter(self._user_cache)))
        self._user_cache[cache_key] = (now, items)
        return items

    async def _list(self, path: str, key: str, api_token: str | None) -> list[dict[str, Any]]:
        if api_token is None:
            return await self._get_cached(path, key)
        return await self._get_for_user(path, key, api_token)

    async def list_datasets(self, *, api_token: str | None = None) -> list[dict[str, Any]]:
        return await self._list("/datasets", "datasets", api_token)

    async def list_collections(self, *, api_token: str | None = None) -> list[dict[str, Any]]:
        return await self._list("/collections", "collections", api_token)

    async def get_dataset(self, slug: str, *, api_token: str | None = None) -> dict[str, Any]:
        for dataset in await self.list_datasets(api_token=api_token):
            if dataset["slug"] == slug:
                return dataset
        raise DatasetNotFoundError(slug)

    async def access(self, slug: str, *, api_token: str) -> dict[str, Any]:
        """The user's credential for one dataset (POST /datasets/{slug}/access).

        Signed cookies arrive as Set-Cookie headers; they are returned under
        "cookies" like the other credential fields.
        """
        resp = await self._client.post(
            f"{self._api_url}/datasets/{slug}/access",
            headers={"Authorization": f"Bearer {api_token}"},
        )
        if resp.status_code in (401, 403, 404):
            raise DatasetAccessDeniedError(slug)
        resp.raise_for_status()
        credential = resp.json()
        if credential.get("credential_type") == "signed_cookies":
            if "cookies" not in credential:
                credential["cookies"] = _cloudfront_cookies(resp)
            missing = [c for c in CLOUDFRONT_COOKIES if not credential["cookies"].get(c)]
            if missing:
                raise ValueError(f"signed-cookie credential has no CloudFront cookies {missing}")
        return credential

    async def aclose(self) -> None:
        await self._client.aclose()
