"""End-to-end equivalence test:

Runs both compare_groups (X-scan) and compare_groups_via_strata against the
strata-tiny.zarr fixture (from packages/zarr-access/tests/fixtures/) and
verifies the gene rankings agree. This is the correctness check that
justifies the upcoming consolidation in a follow-up PR.
"""

import socket
from pathlib import Path

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


@pytest_asyncio.fixture
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


@pytest_asyncio.fixture
async def two_adapters(fixture_server):
    """Open AnnData + Strata adapters against the strata-tiny.zarr fixture."""
    url = f"{fixture_server}/strata-tiny.zarr"
    zarr_store = await ZarrStore.open(url)
    anndata = await AnnDataStore.open(zarr_store)
    strata_store = await StrataStore.open(zarr_store)
    return AnnDataZarrAccess(anndata), StrataZarrAccess(strata_store)


async def test_strata_path_matches_xscan_path_top_ranked_genes(two_adapters):
    """Both tools, identical args, should agree on the TOP gene by absolute LFC.

    Fixture: 50 cells × 10 genes. cell_type A=20, B=20, C=10. Same query
    through both code paths should pick the same most-divergent gene.
    """
    adapter, strata = two_adapters
    config = AgentConfig()
    catalog = build_v1_catalog(adapter, config=config, strata=strata)

    compare_xscan = catalog.get("compare_groups")
    compare_strata = catalog.get("compare_groups_via_strata")
    assert compare_xscan is not None
    assert compare_strata is not None

    args = dict(obs_column="cell_type", group_a="A", group_b="B", n=10)
    result_x = await compare_xscan.func(**args)
    result_s = await compare_strata.func(**args)

    assert "error" not in result_x, result_x
    assert "error" not in result_s, result_s

    # method label distinguishes the two paths
    assert result_x["method"] == "log2_fold_change_fast_heuristic"
    assert result_s["method"] == "log2_fold_change_via_coarse_strata"

    # Both paths agree on the top gene
    assert result_x["genes"][0]["symbol"] == result_s["genes"][0]["symbol"]


async def test_strata_path_returns_clean_error_when_coarse_missing(two_adapters):
    """obs_column 'donor' has no matching coarse table in the fixture
    (the fixture only built coarse for cell_type)."""
    adapter, strata = two_adapters
    config = AgentConfig()
    catalog = build_v1_catalog(adapter, config=config, strata=strata)

    compare_strata = catalog.get("compare_groups_via_strata")
    assert compare_strata is not None

    result = await compare_strata.func(
        obs_column="donor", group_a="d1", group_b="d2"
    )
    assert "error" in result
    assert "no coarse strata table covers axes" in result["error"]
