# cell-explorer-mcp

MCP server exposing [Cell Explorer](https://cell-explorer.cbioportal.org)
single-cell datasets. Dataset and collection listings come from the Cell
Explorer REST API; expression and metadata queries read the public AnnData zarr
stores directly via `zarr-access`.

## Tools

| Tool | What it returns |
|---|---|
| `list_datasets` | Public datasets (slug, name, collection, cell/gene counts); filter by `collection` or text `query` |
| `list_collections` | Dataset collections |
| `get_dataset_info` | Cell/gene counts, obs columns with types and categories, var columns, embeddings, `fast_group_by_columns` |
| `search_genes` | Genes matching a symbol/ID substring (exact > prefix > substring) |
| `gene_expression` | Mean + fraction expressing for one gene, overall and per group of a categorical obs column |
| `top_genes` | Top-N genes by mean expression in one group of cells |
| `obs_column_values` | Cell counts per value of an obs column, or numeric summary stats |

## Data-volume limits

The server is shared and datasets reach millions of cells, so queries are
bounded (see `queries.py`):

- Grouped stats use the precomputed coarse strata tables when one exists for
  the column (`fast_group_by_columns`). No per-cell data is read.
- Per-cell gene reads are limited to datasets of up to 2M cells, at most 2
  concurrently.
- Full-matrix scans (`top_genes` on a column without strata) are limited to
  250M cells × genes.
- There is no differential-expression tool.

## Running

```bash
# stdio (Claude Desktop / Claude Code)
uv run --package cell-explorer-mcp cell-explorer-mcp

# streamable HTTP
MCP_TRANSPORT=http MCP_HOST=0.0.0.0 MCP_PORT=8000 uv run --package cell-explorer-mcp cell-explorer-mcp
```

| Variable | Default |
|---|---|
| `MCP_TRANSPORT` | `stdio` (`http`, `sse`) |
| `MCP_HOST` / `MCP_PORT` / `MCP_HTTP_PATH` | `127.0.0.1` / `8000` / `/mcp` |
| `CELL_EXPLORER_API_URL` | `https://cell-explorer.cbioportal.org/api` |
| `CELL_EXPLORER_MCP_MAX_DATASETS` | `8` (open stores kept in the LRU cache) |

Docker (build from the repo root):

```bash
docker build -f packages/mcp/Dockerfile -t cell-explorer-mcp .
docker run -p 8000:8000 cell-explorer-mcp   # serves http://localhost:8000/mcp
```

## Tests

```bash
uv run pytest packages/mcp/tests/                 # fixture stores served locally
uv run pytest packages/mcp/tests/ -m integration  # live SPECTRUM dataset
```
