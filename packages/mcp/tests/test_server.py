"""End-to-end through the MCP protocol with an in-memory client."""

import os
import subprocess
import sys

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


async def test_list_datasets_reports_collection_slug(client):
    result = (await client.call_tool("list_datasets", {})).data
    by_slug = {d["slug"]: d for d in result["datasets"]}
    assert by_slug["strata-tiny"]["collection"] == "fixtures"
    assert by_slug["pbmc3k"]["collection"] is None


async def test_list_collections(client):
    result = (await client.call_tool("list_collections", {})).data
    assert result["collections"] == [{"slug": "fixtures", "name": "Fixtures", "dataset_count": 1}]


async def test_gene_expression_tool(client):
    result = (await client.call_tool(
        "gene_expression", {"dataset": "strata-tiny", "gene": "gene1", "group_by": "cell_type"}
    )).data
    assert result["source"] == "strata"
    assert {g["group"] for g in result["groups"]} == {"A", "B", "C"}


@pytest.mark.parametrize("tool, arg", [("gene_expression", "gene"), ("search_genes", "query")])
async def test_gene_names_are_limited_to_64_characters(client, tool, arg):
    # Suggestions for an unknown name rescan all genes once per character.
    with pytest.raises(ToolError, match="at most 64 characters"):
        await client.call_tool(tool, {"dataset": "pbmc3k", arg: "Z" * 65})


async def test_unknown_dataset_is_tool_error(client):
    with pytest.raises(ToolError, match="Unknown dataset"):
        await client.call_tool("get_dataset_info", {"dataset": "nope"})


async def test_private_dataset_is_refused(client):
    with pytest.raises(ToolError, match="not public"):
        await client.call_tool("get_dataset_info", {"dataset": "secret"})


async def test_obs_column_values_accepts_limit_500(client):
    result = (await client.call_tool(
        "obs_column_values", {"dataset": "pbmc3k", "column": "louvain", "limit": 500}
    )).data
    assert result["n_values"] == 8


async def test_query_error_is_tool_error(client):
    with pytest.raises(ToolError, match="not found"):
        await client.call_tool("obs_column_values", {"dataset": "pbmc3k", "column": "nope"})


def test_instructions_link_to_the_configured_cell_explorer():
    # The URL is read at import, so check it in a fresh interpreter.
    env = {**os.environ, "CELL_EXPLORER_URL": "https://btc.example.org"}
    code = "from cell_explorer_mcp import server; print(server.mcp.instructions)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert "(https://btc.example.org)" in out
    assert "(https://btc.example.org/view?dataset=<slug>)" in out
    assert "cbioportal.org" not in out


def test_instructions_explain_scaled_data_and_differential_expression():
    text = " ".join(server.mcp.instructions.split())
    assert '"fraction_expressing" means the fraction of cells above average' in text
    assert "on scaled data it approximates the group's markers versus all cells" in text
    assert "It does not compare two groups." in text
    assert "use gene_expression with group_by (means and fractions, no statistics)" in text
    assert "where the chat assistant can compare groups if chat is enabled for that dataset" in text


async def test_top_genes_description_is_not_a_group_comparison(client):
    tools = {t.name: t for t in await client.list_tools()}
    description = " ".join(tools["top_genes"].description.split())
    assert "on scaled data the ranking approximates the group's markers versus all cells" in description
    assert "Not a comparison between two groups." in description
