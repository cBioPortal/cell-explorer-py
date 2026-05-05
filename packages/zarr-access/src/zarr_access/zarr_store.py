"""Async zarr store wrapper with optional bearer token auth."""

from typing import Literal

import aiohttp
import zarr
import zarr.api.asynchronous
from zarr.storage import FsspecStore


class ZarrStore:
    """Async zarr store wrapper over HTTP.

    Wraps zarr-python 3.x with fsspec async HTTP filesystem. Auto-detects
    zarr v2 vs v3 by probing for zarr.json (v3) then .zmetadata (v2).
    Optional bearer token support via the `headers` argument to `open`.
    """

    def __init__(
        self,
        root_group: zarr.AsyncGroup,
        zarr_version: Literal[2, 3],
        consolidated_metadata: dict | None = None,
    ):
        self._root = root_group
        self._zarr_version = zarr_version
        self._consolidated_metadata = consolidated_metadata

    @property
    def zarr_version(self) -> Literal[2, 3]:
        return self._zarr_version

    @property
    def consolidated_metadata(self) -> dict | None:
        return self._consolidated_metadata

    @classmethod
    async def open(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> "ZarrStore":
        """Open a zarr store over HTTP.

        Args:
            url: Full URL to the zarr store (e.g., http://host/path/to/store.zarr).
            headers: Optional HTTP headers (e.g., {"Authorization": "Bearer <jwt>"}).

        Returns:
            A ZarrStore instance with zarr version auto-detected.
        """
        # Detect version and read consolidated metadata via aiohttp probe
        zarr_version, consolidated = await cls._probe_version(url, headers)

        # Build fsspec store via from_url, passing headers + optional trace
        # configs as client_kwargs. Tracing is opt-in via ZARR_ACCESS_TRACE=1.
        from zarr_access.tracing import build_trace_configs

        client_kwargs: dict = {}
        if headers:
            client_kwargs["headers"] = headers
        trace_configs = build_trace_configs()
        if trace_configs is not None:
            client_kwargs["trace_configs"] = trace_configs

        storage_options: dict = (
            {"client_kwargs": client_kwargs} if client_kwargs else {}
        )

        store = FsspecStore.from_url(url, storage_options=storage_options, read_only=True)

        # Open as appropriate zarr format
        root = await zarr.api.asynchronous.open_group(
            store,
            mode="r",
            zarr_format=zarr_version,
        )
        return cls(root, zarr_version, consolidated)

    @staticmethod
    async def _probe_version(
        url: str,
        headers: dict[str, str] | None,
    ) -> tuple[Literal[2, 3], dict | None]:
        """Probe URL to detect zarr version and consolidated metadata.

        Tries zarr.json (v3) first, then .zmetadata (v2). Returns (version, consolidated_metadata).
        """
        async with aiohttp.ClientSession(headers=headers or {}) as session:
            # Try v3 first
            async with session.get(f"{url}/zarr.json") as resp:
                if resp.status == 200:
                    root_json = await resp.json(content_type=None)
                    zarr_format = root_json.get("zarr_format")
                    consolidated = None
                    if "consolidated_metadata" in root_json:
                        consolidated = root_json["consolidated_metadata"].get("metadata")
                    version: Literal[2, 3] = 3 if zarr_format == 3 else 2
                    return version, consolidated

            # Fall back to v2
            async with session.get(f"{url}/.zmetadata") as zmeta_resp:
                if zmeta_resp.status == 200:
                    zmeta = await zmeta_resp.json(content_type=None)
                    return 2, zmeta.get("metadata")

        # Couldn't determine — default to v2 and no metadata
        return 2, None

    async def get_array(self, path: str) -> zarr.AsyncArray:
        """Get a zarr array at the given path within the store."""
        return await self._root.getitem(path)

    async def get_group(self, path: str) -> zarr.AsyncGroup:
        """Get a zarr group at the given path within the store."""
        return await self._root.getitem(path)
