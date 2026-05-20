"""End-to-end test of find_markers against strata-tiny.zarr.

The fixture is built by zarr-access's test fixtures — 50 cells x 10 genes
with a coarse strata table on cell_type and an atomic table on
['cell_type', 'donor']. Verifies that:
  - The coarse strata fast path runs when the obs_column is covered.
  - The atomic fallback runs when only atomic covers obs_column.
  - Both paths agree with X-scan on per-gene stats for the same query.
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


async def test_find_markers_uses_coarse_strata_on_fixture(adapters):
    """cell_type is covered by the fixture's coarse table -> via_coarse_strata."""
    adapter, strata = adapters
    catalog = build_v1_catalog(adapter, config=AgentConfig(), strata=strata)
    tool = catalog.get("find_markers")
    assert tool is not None

    result = await tool.func(obs_column="cell_type", group_value="A", n=5)
    assert "error" not in result, result
    assert result["method"] == "via_coarse_strata"
    assert result["n_cells_group"] > 0
    assert result["n_cells_rest"] > 0
    # group + rest must partition the dataset (no overlap, no miss).
    assert result["n_cells_group"] + result["n_cells_rest"] == 50
    assert len(result["genes"]) <= 5


async def test_find_markers_uses_atomic_when_donor_only_in_atomic(adapters):
    """donor isn't in coarse but is one of the atomic axes -> via_atomic_strata."""
    adapter, strata = adapters
    catalog = build_v1_catalog(adapter, config=AgentConfig(), strata=strata)
    tool = catalog.get("find_markers")
    assert tool is not None

    result = await tool.func(obs_column="donor", group_value="d1", n=5)
    assert "error" not in result, result
    assert result["method"] == "via_atomic_strata"
    assert result["n_cells_group"] + result["n_cells_rest"] == 50


async def test_find_markers_paths_agree_on_fixture(adapters):
    """Strata and X-scan paths must agree on per-gene stats and ranking."""
    adapter, strata = adapters

    catalog_strata = build_v1_catalog(adapter, config=AgentConfig(), strata=strata)
    catalog_xscan = build_v1_catalog(adapter, config=AgentConfig(), strata=None)

    args = dict(obs_column="cell_type", group_value="A", n=5)
    result_strata = await catalog_strata.get("find_markers").func(**args)
    result_xscan = await catalog_xscan.get("find_markers").func(**args)

    assert result_strata["method"] == "via_coarse_strata"
    assert result_xscan["method"] == "via_xscan"
    assert result_strata["n_cells_group"] == result_xscan["n_cells_group"]
    assert result_strata["n_cells_rest"] == result_xscan["n_cells_rest"]
    # Same top-ranked gene.
    assert result_strata["genes"][0]["symbol"] == result_xscan["genes"][0]["symbol"]
    # Per-gene stats agree (modulo float precision between float32 strata
    # sums and float64 X-scan accumulators).
    for gs, gx in zip(result_strata["genes"], result_xscan["genes"]):
        np.testing.assert_allclose(gs["cohens_d"], gx["cohens_d"], rtol=1e-4)
        np.testing.assert_allclose(gs["t_statistic"], gx["t_statistic"], rtol=1e-4)
