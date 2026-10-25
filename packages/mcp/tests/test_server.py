"""End-to-end through the MCP protocol with an in-memory client."""

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError

from cell_explorer_mcp import server
from cell_explorer_mcp.zarr_cache import DatasetCache

from conftest import stub_catalog


@pytest_asyncio.fixture(loop_scope="session")
async def client(fixture_server, monkeypatch):
    monkeypatch.setattr(server, "catalog", stub_catalog(fixture_server))
    monkeypatch.setattr(server, "cache", DatasetCache())
    async with Client(server.mcp) as c:
        yield c


async def test_lists_all_tools(client):
    names = {t.name for t in await client.list_tools()}
    assert names == {
        "list_datasets",
        "list_collections",
        "get_dataset_info",
        "search_genes",
        "gene_expression",
        "top_genes",
        "obs_column_values",
    }


async def test_list_datasets_hides_private_and_filters(client):
    result = (await client.call_tool("list_datasets", {})).data
    assert {d["slug"] for d in result["datasets"]} == {"strata-tiny", "pbmc3k"}

    by_collection = (await client.call_tool("list_datasets", {"collection": "fixtures"})).data
    assert [d["slug"] for d in by_collection["datasets"]] == ["strata-tiny"]

    by_text = (await client.call_tool("list_datasets", {"query": "mononuclear"})).data
    assert [d["slug"] for d in by_text["datasets"]] == ["pbmc3k"]


async def test_list_collections(client):
    result = (await client.call_tool("list_collections", {})).data
    assert result["collections"] == [{"slug": "fixtures", "name": "Fixtures", "dataset_count": 1}]


async def test_gene_expression_tool(client):
    result = (await client.call_tool(
        "gene_expression", {"dataset": "strata-tiny", "gene": "gene1", "group_by": "cell_type"}
    )).data
    assert result["source"] == "strata"
    assert {g["group"] for g in result["groups"]} == {"A", "B", "C"}


async def test_unknown_dataset_is_tool_error(client):
    with pytest.raises(ToolError, match="Unknown dataset"):
        await client.call_tool("get_dataset_info", {"dataset": "nope"})


async def test_private_dataset_is_refused(client):
    with pytest.raises(ToolError, match="not public"):
        await client.call_tool("get_dataset_info", {"dataset": "secret"})


async def test_query_error_is_tool_error(client):
    with pytest.raises(ToolError, match="not found"):
        await client.call_tool("obs_column_values", {"dataset": "pbmc3k", "column": "nope"})
