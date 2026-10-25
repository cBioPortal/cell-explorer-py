"""Fixtures: zarr-access's test stores served over HTTP, plus a stub catalog."""

import socket
from pathlib import Path

import anndata as ad
import httpx
import pytest
import pytest_asyncio
from aiohttp import web

from cell_explorer_mcp.catalog import CatalogClient
from cell_explorer_mcp.zarr_cache import open_dataset

FIXTURES_DIR = Path(__file__).parents[2] / "zarr-access" / "tests" / "fixtures"


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def fixture_server():
    app = web.Application()
    app.router.add_static("/", FIXTURES_DIR)
    port = _find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def strata_tiny(fixture_server):
    """50 cells x 10 genes, dense X, coarse strata on cell_type only."""
    return await open_dataset("strata-tiny", f"{fixture_server}/strata-tiny.zarr")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pbmc(fixture_server):
    """2638 cells x 1838 genes, dense X, no strata."""
    return await open_dataset("pbmc3k", f"{fixture_server}/pbmc3k_v3.zarr")


@pytest.fixture(scope="session")
def strata_tiny_adata():
    return ad.read_zarr(FIXTURES_DIR / "strata-tiny.zarr")


@pytest.fixture(scope="session")
def pbmc_adata():
    return ad.read_zarr(FIXTURES_DIR / "pbmc3k_v3.zarr")


def catalog_payload(base_url: str) -> dict:
    return {
        "datasets": [
            {
                "slug": "strata-tiny",
                "name": "Strata tiny",
                "description": "Synthetic fixture",
                "is_public": True,
                "url": f"{base_url}/strata-tiny.zarr",
                "collection": "fixtures",
                "metadata": {"n_obs": 50, "n_vars": 10},
            },
            {
                "slug": "pbmc3k",
                "name": "PBMC 3k",
                "description": "Peripheral blood mononuclear cells",
                "is_public": True,
                "url": f"{base_url}/pbmc3k_v3.zarr",
                "collection": None,
                "metadata": {"n_obs": 2638, "n_vars": 1838},
            },
            {
                "slug": "secret",
                "name": "Private",
                "description": "",
                "is_public": False,
                "url": f"{base_url}/strata-tiny.zarr",
                "collection": None,
            },
        ],
        "collections": [{"slug": "fixtures", "name": "Fixtures", "dataset_count": 1}],
    }


def stub_catalog(base_url: str, calls: list[str] | None = None, **kwargs) -> CatalogClient:
    payload = catalog_payload(base_url)

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        key = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={key: payload[key]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return CatalogClient("http://catalog.test/api", client=client, **kwargs)
