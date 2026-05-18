"""End-to-end test of the unified compare_groups against strata-tiny.zarr.

The fixture is built by zarr-access's test fixtures — 50 cells x 10 genes
with a coarse strata table on cell_type. Verifies that:
  - The strata fast path runs when the obs_column is covered.
  - X-scan fallback runs when it isn't.
  - Both paths agree on per-gene stats for the same query.
"""

import socket
from pathlib import Path

import numpy as np
import pytest_asyncio
from aiohttp import web

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.tools import build_v1_catalog
from cell_explorer_api.services.zarr_adapter import (
    AnnDataZarrAccess,
    StrataZarrAccess,
)
from zarr_access import AnnDataStore, StrataStore, ZarrStore


FIXTURE_DIR = (
    Path(__file__).parent.parent.parent
    / "zarr-access" / "tests" / "fixtures"
)


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture(scope="module")
async def fixture_server():
    """Serve packages/zarr-access/tests/fixtures over HTTP on a random port."""
    app = web.Application()
    app.router.add_static("/", FIXTURE_DIR, show_index=True)
    port = _find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@pytest_asyncio.fixture(scope="module")
async def adapters(fixture_server):
    """Open AnnData + Strata adapters against the strata-tiny.zarr fixture."""
    url = f"{fixture_server}/strata-tiny.zarr"
    zarr_store = await ZarrStore.open(url)
    anndata = await AnnDataStore.open(zarr_store)
    strata_store = await StrataStore.open(zarr_store)
    return AnnDataZarrAccess(anndata), StrataZarrAccess(strata_store)


async def test_compare_groups_uses_strata_on_fixture(adapters):
    """cell_type is covered by the fixture's coarse table -> via_coarse_strata."""
    adapter, strata = adapters
    catalog = build_v1_catalog(adapter, config=AgentConfig(), strata=strata)
    tool = catalog.get("compare_groups")
    assert tool is not None

    result = await tool.func(
        obs_column="cell_type", group_a="A", group_b="B", n=5
    )
    assert "error" not in result, result
    assert result["method"] == "via_coarse_strata"
    assert len(result["genes"]) <= 5


async def test_compare_groups_falls_back_when_obs_column_not_in_strata(adapters):
    """'donor' is in the fixture's obs but has no coarse strata table -> via_xscan.

    The fixture carries 'cell_type' (covered by a coarse table) and 'donor'
    (not covered). Querying on 'donor' should skip the strata fast-path and
    fall back to the X-scan path. If the fixture is ever extended to add a
    coarse table for 'donor', this test should be updated accordingly.
    """
    adapter, strata = adapters
    catalog = build_v1_catalog(adapter, config=AgentConfig(), strata=strata)
    tool = catalog.get("compare_groups")
    assert tool is not None

    result = await tool.func(
        obs_column="donor", group_a="d1", group_b="d2"
    )
    assert "error" not in result, result
    assert result["method"] == "via_xscan", result


async def test_compare_groups_paths_agree_on_fixture(adapters):
    """Strata and X-scan agree on top-ranked gene for same query."""
    adapter, strata = adapters

    # Same call, but force X-scan by passing strata=None to a fresh catalog.
    catalog_strata = build_v1_catalog(adapter, config=AgentConfig(), strata=strata)
    catalog_xscan = build_v1_catalog(adapter, config=AgentConfig(), strata=None)

    args = dict(obs_column="cell_type", group_a="A", group_b="B", n=5)
    result_strata = await catalog_strata.get("compare_groups").func(**args)
    result_xscan = await catalog_xscan.get("compare_groups").func(**args)

    assert result_strata["method"] == "via_coarse_strata"
    assert result_xscan["method"] == "via_xscan"
    assert result_strata["genes"][0]["symbol"] == result_xscan["genes"][0]["symbol"]
    # Per-gene stats agree (modulo float precision between float32 strata sums
    # and float64 X-scan accumulators).
    for gs, gx in zip(result_strata["genes"], result_xscan["genes"]):
        np.testing.assert_allclose(gs["cohens_d"], gx["cohens_d"], rtol=1e-4)
        np.testing.assert_allclose(gs["t_statistic"], gx["t_statistic"], rtol=1e-4)
