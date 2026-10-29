"""Live checks against cell-explorer.cbioportal.org. Run with `-m integration`."""

import pytest
from fastmcp import Client

from cell_explorer_mcp import server
from cell_explorer_mcp.catalog import CatalogClient
from cell_explorer_mcp.zarr_cache import DatasetCache

pytestmark = pytest.mark.integration


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(server, "catalog", CatalogClient())
    monkeypatch.setattr(server, "cache", DatasetCache())


async def test_spectrum_cd8a_by_cell_type(live):
    async with Client(server.mcp) as client:
        datasets = (await client.call_tool("list_datasets", {"query": "spectrum"})).data
        assert "spectrum" in {d["slug"] for d in datasets["datasets"]}

        result = (await client.call_tool(
            "gene_expression",
            {"dataset": "spectrum", "gene": "CD8A", "group_by": "author_cell_type"},
        )).data
        assert result["overall"]["n_cells"] > 900_000
        assert result["groups"][0]["group"] == "T.cell"


async def test_spectrum_top_t_cell_genes(live):
    async with Client(server.mcp) as client:
        result = (await client.call_tool(
            "top_genes",
            {"dataset": "spectrum", "obs_column": "author_cell_type", "group_value": "T.cell", "n": 10},
        )).data
        assert result["source"] == "strata"
        assert {"CD3D", "CD3E"} & {g["symbol"] for g in result["genes"]}
