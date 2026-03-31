# cell-explorer-py

Python backend for [cBioPortal Cell Explorer](https://github.com/cBioPortal/cbioportal-cell-explorer). Converts single-cell RNA-seq data (h5ad) to Zarr v3 stores optimized for web-based visualization.

## Packages

This is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo:

| Package | Description |
|---------|-------------|
| `cell2zarr` | h5ad to Zarr conversion pipeline |
| `cell-explorer-core` | Shared config and settings |
| `cell-explorer-auth` | Keycloak OAuth2 + CloudFront signed cookies |
| `cell-explorer-api` | FastAPI API + static file serving |

## Setup

```bash
uv sync
```

## cell2zarr CLI

### Convert h5ad to Zarr

```bash
# Basic conversion
cell2zarr convert input.h5ad output.zarr

# Two-phase streaming conversion (low memory, recommended for large datasets)
cell2zarr convert input.h5ad output.zarr --two-phase --encoding-config encoding.json

# With sharding and custom dtype
cell2zarr convert input.h5ad output.zarr --two-phase --shard-size 30 --dtype float16
```

The default command is `convert`, so `cell2zarr input.h5ad output.zarr` also works.

### Add keys to existing Zarr store

Add individual keys from an h5ad file to an existing Zarr store without re-running the full conversion:

```bash
# Add a specific obsm embedding
cell2zarr add atlas.h5ad atlas.zarr --key obsm/X_umap

# Add all obsm keys
cell2zarr add atlas.h5ad atlas.zarr --key obsm

# Add obs annotations (overwrite existing)
cell2zarr add atlas.h5ad atlas.zarr --key obs --overwrite

# Add expression matrix (uses two-phase pipeline)
cell2zarr add atlas.h5ad atlas.zarr --key X --temp-dir /tmp

# Add a specific layer
cell2zarr add atlas.h5ad atlas.zarr --key layers/counts --temp-dir /tmp
```

Supported keys: `obsm`, `obs`, `var`, `uns`, `obsp`, `varp`, `X`, `layers`.

### Utility scripts

```bash
# Inspect h5ad file structure and X preprocessing state
uv run python scripts/inspect_h5ad.py input.h5ad

# Generate UMAP embedding
uv run python scripts/generate_umap.py input.h5ad output_with_umap.h5ad
```

## Tests

```bash
uv run pytest packages/cell2zarr/tests/ -v
```

## License

MIT
