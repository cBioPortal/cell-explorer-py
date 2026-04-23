"""Pytest fixtures for zarr-access tests."""

import socket
from pathlib import Path

import pytest_asyncio
from aiohttp import web


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def fixture_server():
    """Serve the fixtures directory over HTTP on a random port."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    app = web.Application()
    app.router.add_static("/", fixtures_dir, show_index=True)

    port = _find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    await runner.cleanup()


AUTH_TOKEN = "test-bearer-token-value"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def auth_fixture_server():
    """Serve fixtures/ over HTTP, requiring Authorization: Bearer AUTH_TOKEN header."""
    fixtures_dir = Path(__file__).parent / "fixtures"

    @web.middleware
    async def auth_middleware(request, handler):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {AUTH_TOKEN}":
            return web.Response(status=401, text="Missing or invalid token")
        return await handler(request)

    app = web.Application(middlewares=[auth_middleware])
    app.router.add_static("/", fixtures_dir, show_index=True)

    port = _find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url, AUTH_TOKEN
    await runner.cleanup()
