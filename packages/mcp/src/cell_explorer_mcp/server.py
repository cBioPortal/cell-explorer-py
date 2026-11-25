"""Cell Explorer MCP server: single-cell dataset queries over the Model Context Protocol."""

import logging
import os
from typing import Annotated, Any

import fastmcp
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AuthProvider
from pydantic import Field

from cell_explorer_mcp import queries
from cell_explorer_mcp.catalog import DEFAULT_API_URL, CatalogClient, DatasetNotFoundError
from cell_explorer_mcp.zarr_cache import DEFAULT_MAX_DATASETS, DatasetCache, OpenDataset

logger = logging.getLogger(__name__)

# Links follow CELL_EXPLORER_URL, so each instance points at its own website.
INSTRUCTIONS = f"""\
Query single-cell RNA-seq datasets hosted on Cell Explorer
({queries.UI_URL}).

Workflow:
1. list_datasets (optionally filtered by collection or text) to find a dataset slug.
2. get_dataset_info to learn its cell metadata columns (obs columns), their
   number of distinct values, and which of them have precomputed group
   statistics (fast_group_by_columns).
3. obs_column_values to get a column's exact values before using one as
   group_value; naming varies by dataset (e.g. 'T.cell' vs 'T cell').
4. search_genes to resolve gene names; then gene_expression or top_genes.

Expression values are as stored in each dataset: usually log-normalized; some
datasets are scaled (each gene z-scored across cells), where a negative mean is
below that gene's dataset average and "fraction_expressing" means the fraction
of cells above average rather than expressing.
Grouping by a column in fast_group_by_columns is cheap; other columns read
per-cell data and may be refused on very large datasets.
top_genes ranks genes by mean within one group: on log-normalized data broadly
expressed genes (ribosomal, mitochondrial, MALAT1, B2M) lead; on scaled data it
approximates the group's markers versus all cells. It does not compare two
groups. Differential expression is not offered: to compare candidate genes
across groups use gene_expression with group_by (means and fractions, no
statistics), and for a marker or differential analysis point users to the
dataset page ({queries.UI_URL}/view?dataset=<slug>), where the chat assistant
can compare groups if chat is enabled for that dataset.
"""

catalog = CatalogClient(os.getenv("CELL_EXPLORER_API_URL", DEFAULT_API_URL))
cache = DatasetCache(maxsize=int(os.getenv("CELL_EXPLORER_MCP_MAX_DATASETS", DEFAULT_MAX_DATASETS)))

# Gene symbols and IDs (including versioned Ensembl IDs) are far shorter; the
# limit bounds the per-character rescan behind unknown-gene suggestions.
MAX_GENE_NAME = 64

DatasetSlug = Annotated[str, Field(description="Dataset slug from list_datasets, e.g. 'spectrum'.")]


async def _dataset_entry(slug: str) -> dict[str, Any]:
    try:
        entry = await catalog.get_dataset(slug)
    except DatasetNotFoundError:
        raise ToolError(f"Unknown dataset {slug!r}. Use list_datasets to find valid slugs.") from None
    if not entry.get("is_public", False):
        raise ToolError(f"Dataset {slug!r} is not public.")
    return entry


async def _open(slug: str) -> tuple[OpenDataset, dict[str, Any]]:
    entry = await _dataset_entry(slug)
    try:
        ds = await cache.get(slug, entry["url"])
    except Exception as exc:
        logger.exception("failed to open dataset %s", slug)
        raise ToolError(f"Could not open dataset {slug!r}: {exc}") from exc
    return ds, entry


async def _run(coro):
    try:
        return await coro
    except queries.QueryError as exc:
        raise ToolError(str(exc)) from exc


def _collection_slug(dataset: dict[str, Any]) -> str | None:
    # The API returns the collection as an object (DatasetCollectionRef) or null.
    return (dataset.get("collection") or {}).get("slug")


async def list_datasets(
    collection: Annotated[str | None, Field(description="Only datasets in this collection slug.")] = None,
    query: Annotated[str | None, Field(description="Case-insensitive text filter on slug, name and description.")] = None,
) -> dict[str, Any]:
    """List public Cell Explorer datasets with their size and collection."""
    datasets = await catalog.list_datasets()
    q = query.lower() if query else None
    out = []
    for d in datasets:
        if not d.get("is_public", False):
            continue
        if collection is not None and _collection_slug(d) != collection:
            continue
        text = " ".join(str(d.get(k) or "") for k in ("slug", "name", "description")).lower()
        if q is not None and q not in text:
            continue
        metadata = d.get("metadata") or {}
        out.append({
            "slug": d["slug"],
            "name": d.get("name"),
            "description": d.get("description"),
            # Slug only; list_collections and get_dataset_info carry the details.
            "collection": _collection_slug(d),
            "n_cells": metadata.get("n_obs"),
            "n_genes": metadata.get("n_vars"),
        })
    return {"n_datasets": len(out), "datasets": out}


async def list_collections() -> dict[str, Any]:
    """List dataset collections (groups of related datasets, e.g. from one study)."""
    collections = await catalog.list_collections()
    keys = ("slug", "name", "description", "dataset_count", "publication_citation",
            "publication_url", "source_url", "license_name")
    return {
        "n_collections": len(collections),
        "collections": [{k: c.get(k) for k in keys if c.get(k) is not None} for c in collections],
    }


async def get_dataset_info(dataset: DatasetSlug) -> dict[str, Any]:
    """Dataset overview: cell and gene counts, obs (cell metadata) columns with
    types and number of distinct values, var columns, embeddings, and the
    columns with precomputed group statistics. Use obs_column_values for a
    column's values."""
    ds, entry = await _open(dataset)
    return await _run(queries.dataset_info(ds, entry))


async def search_genes(
    dataset: DatasetSlug,
    query: Annotated[str, Field(description="Gene symbol or ID substring, case-insensitive.", max_length=MAX_GENE_NAME)],
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
) -> dict[str, Any]:
    """Find genes by symbol or ID. Exact matches rank first, then prefix, then substring."""
    ds, _ = await _open(dataset)
    return await _run(queries.search_genes(ds, query, limit))


async def gene_expression(
    dataset: DatasetSlug,
    gene: Annotated[str, Field(description="Gene symbol or ID.", max_length=MAX_GENE_NAME)],
    group_by: Annotated[str | None, Field(description="Categorical obs column to break results down by, e.g. a cell type column.")] = None,
) -> dict[str, Any]:
    """Mean expression and fraction of expressing cells for one gene, overall and
    optionally per group of a categorical obs column (top 50 groups by mean)."""
    ds, entry = await _open(dataset)
    return await _run(queries.gene_expression(ds, gene, group_by, entry=entry))


async def top_genes(
    dataset: DatasetSlug,
    obs_column: Annotated[str, Field(description="Categorical obs column, e.g. a cell type column.")],
    group_value: Annotated[str, Field(description="Value of obs_column selecting the cells, e.g. 'T.cell'.")],
    n: Annotated[int, Field(ge=1, le=50)] = 20,
) -> dict[str, Any]:
    """Top-N genes by mean expression within one group of cells. On
    log-normalized data broadly expressed genes (ribosomal, mitochondrial)
    lead; on scaled data the ranking approximates the group's markers versus
    all cells. Not a comparison between two groups."""
    ds, entry = await _open(dataset)
    return await _run(queries.top_genes(ds, obs_column, group_value, n, entry=entry))


async def obs_column_values(
    dataset: DatasetSlug,
    column: Annotated[str, Field(description="Obs (cell metadata) column name.")],
    limit: Annotated[int, Field(ge=1, le=queries.MAX_OBS_VALUES)] = 50,
) -> dict[str, Any]:
    """Cell counts per value of an obs column (most frequent first), or summary
    statistics for a numeric column."""
    ds, entry = await _open(dataset)
    return await _run(queries.obs_column_values(ds, column, limit, entry=entry))


TOOLS = (
    list_datasets,
    list_collections,
    get_dataset_info,
    search_genes,
    gene_expression,
    top_genes,
    obs_column_values,
)


def build_server(*, signed_in: bool, auth: AuthProvider | None = None) -> FastMCP:
    """One MCP server; the signed-in one requires a Cell Explorer sign-in."""
    name = "Cell Explorer MCP Server (signed in)" if signed_in else "Cell Explorer MCP Server"
    server = FastMCP(name=name, instructions=INSTRUCTIONS, auth=auth)
    for tool in TOOLS:
        server.tool(tool)
    return server


# The open server: stdio, and the open HTTP address.
mcp = build_server(signed_in=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run()
        return
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    if transport == "sse":
        # Legacy transport: the open address only.
        mcp.run(transport=transport, host=host, port=port, path=os.getenv("MCP_HTTP_PATH", "/mcp"))
        return
    if transport not in ("http", "streamable-http"):
        raise ValueError(f"Unknown MCP_TRANSPORT {transport!r}")
    import uvicorn

    from cell_explorer_mcp.app import build_http_app

    uvicorn.run(
        build_http_app(),
        host=host,
        port=port,
        timeout_graceful_shutdown=2,
        log_level=fastmcp.settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
