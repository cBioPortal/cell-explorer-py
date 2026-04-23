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
