"""Cell Explorer MCP server: single-cell dataset queries over the Model Context Protocol."""

import logging
import os
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from cell_explorer_mcp import queries
from cell_explorer_mcp.catalog import DEFAULT_API_URL, CatalogClient, DatasetNotFoundError
from cell_explorer_mcp.zarr_cache import DEFAULT_MAX_DATASETS, DatasetCache, OpenDataset

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Query single-cell RNA-seq datasets hosted on Cell Explorer
(https://cell-explorer.cbioportal.org).

Workflow:
1. list_datasets (optionally filtered by collection or text) to find a dataset slug.
2. get_dataset_info to learn its cell metadata columns (obs columns) and which of
   them have precomputed group statistics (fast_group_by_columns).
3. search_genes to resolve gene names; then gene_expression or top_genes.

Expression values are as stored in each dataset: usually log-normalized, but
some datasets are scaled (z-scored), where negative means are normal.
"fraction_expressing" is the fraction of cells with a value > 0.
Grouping by a column in fast_group_by_columns is cheap; other columns read
per-cell data and may be refused on very large datasets. Differential
expression between groups is not offered; point users to the Cell Explorer UI
(https://cell-explorer.cbioportal.org/<slug>) for that.
"""

catalog = CatalogClient(os.getenv("CELL_EXPLORER_API_URL", DEFAULT_API_URL))
cache = DatasetCache(maxsize=int(os.getenv("CELL_EXPLORER_MCP_MAX_DATASETS", DEFAULT_MAX_DATASETS)))

mcp = FastMCP(name="Cell Explorer MCP Server", instructions=INSTRUCTIONS)

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


@mcp.tool()
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
        if collection is not None and d.get("collection") != collection:
            continue
        text = " ".join(str(d.get(k) or "") for k in ("slug", "name", "description")).lower()
        if q is not None and q not in text:
            continue
        metadata = d.get("metadata") or {}
        out.append({
            "slug": d["slug"],
            "name": d.get("name"),
            "description": d.get("description"),
            "collection": d.get("collection"),
            "n_cells": metadata.get("n_obs"),
            "n_genes": metadata.get("n_vars"),
        })
    return {"n_datasets": len(out), "datasets": out}


@mcp.tool()
async def list_collections() -> dict[str, Any]:
    """List dataset collections (groups of related datasets, e.g. from one study)."""
    collections = await catalog.list_collections()
    keys = ("slug", "name", "description", "dataset_count", "publication_citation",
            "publication_url", "source_url", "license_name")
    return {
        "n_collections": len(collections),
        "collections": [{k: c.get(k) for k in keys if c.get(k) is not None} for c in collections],
    }


@mcp.tool()
async def get_dataset_info(dataset: DatasetSlug) -> dict[str, Any]:
    """Dataset overview: cell and gene counts, obs (cell metadata) columns with
    types and categories, var columns, embeddings, and the columns with
    precomputed group statistics."""
    ds, entry = await _open(dataset)
    return await _run(queries.dataset_info(ds, entry))


@mcp.tool()
async def search_genes(
    dataset: DatasetSlug,
    query: Annotated[str, Field(description="Gene symbol or ID substring, case-insensitive.")],
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
) -> dict[str, Any]:
    """Find genes by symbol or ID. Exact matches rank first, then prefix, then substring."""
    ds, _ = await _open(dataset)
    return await _run(queries.search_genes(ds, query, limit))


@mcp.tool()
async def gene_expression(
    dataset: DatasetSlug,
    gene: Annotated[str, Field(description="Gene symbol or ID.")],
    group_by: Annotated[str | None, Field(description="Categorical obs column to break results down by, e.g. a cell type column.")] = None,
) -> dict[str, Any]:
    """Mean expression and fraction of expressing cells for one gene, overall and
    optionally per group of a categorical obs column (top 50 groups by mean)."""
    ds, _ = await _open(dataset)
    return await _run(queries.gene_expression(ds, gene, group_by))


@mcp.tool()
async def top_genes(
    dataset: DatasetSlug,
    obs_column: Annotated[str, Field(description="Categorical obs column, e.g. a cell type column.")],
    group_value: Annotated[str, Field(description="Value of obs_column selecting the cells, e.g. 'T.cell'.")],
    n: Annotated[int, Field(ge=1, le=50)] = 20,
) -> dict[str, Any]:
    """Top-N genes by mean expression within one group of cells. Ranks by
    absolute expression, not specificity: housekeeping and ribosomal genes
    often lead."""
    ds, _ = await _open(dataset)
    return await _run(queries.top_genes(ds, obs_column, group_value, n))


@mcp.tool()
async def obs_column_values(
    dataset: DatasetSlug,
    column: Annotated[str, Field(description="Obs (cell metadata) column name.")],
    limit: Annotated[int, Field(ge=1, le=50)] = 50,
) -> dict[str, Any]:
    """Cell counts per value of an obs column (most frequent first), or summary
    statistics for a numeric column."""
    ds, _ = await _open(dataset)
    return await _run(queries.obs_column_values(ds, column, limit))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run()
        return
    mcp.run(
        transport=transport,
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8000")),
        path=os.getenv("MCP_HTTP_PATH", "/mcp"),
    )


if __name__ == "__main__":
    main()
