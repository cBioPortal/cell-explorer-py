# cell-explorer-py

Python backend for [cBioPortal Cell Explorer](https://github.com/cBioPortal/cbioportal-cell-explorer). Converts single-cell RNA-seq data (h5ad) to Zarr v3 stores optimized for web-based visualization.

## Packages

This is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo:

| Package | Description |
|---------|-------------|
| [`cell2zarr`](packages/cell2zarr/) | h5ad to Zarr conversion pipeline |
| `cell-explorer-core` | Shared config and settings |
| `cell-explorer-auth` | Keycloak OAuth2 + CloudFront signed cookies |
| `cell-explorer-api` | FastAPI API + static file serving |

## Setup

```bash
uv sync
```

## Preparing your dataset

To convert single-cell RNA-seq data for use with Cell Explorer, see the [cell2zarr documentation](packages/cell2zarr/README.md).

Quick start:

```bash
# Convert h5ad to Zarr
cell2zarr convert input.h5ad output.zarr --two-phase --encoding-config encoding.json

# Add a UMAP embedding to an existing store
cell2zarr add atlas.h5ad atlas.zarr --key obsm/X_umap
```

## Tests

```bash
uv run pytest packages/cell2zarr/tests/ -v
```

## License

MIT
