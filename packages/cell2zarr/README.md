# cell2zarr

h5ad to Zarr v3 conversion pipeline for single-cell RNA-seq data. Optimized for web-based visualization with column-oriented chunking and sharding.

## Install

```bash
uv sync
```

## CLI

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

### Logging

Every command logs to both stdout and a log file:

```bash
# Default log file: output.zarr.log (for convert) or atlas.zarr.add.log (for add)
cell2zarr convert input.h5ad output.zarr --two-phase

# Custom log file and log level
cell2zarr convert input.h5ad output.zarr --two-phase --log-file /path/to/run.log --log-level DEBUG
```

Log format: `[2026-03-31 10:15:23] INFO: Phase 1 complete in 4863s`

### Run history

Track conversion runs across time with `--run-db`. This writes a JSON file with config, performance metrics, and status for each run:

```bash
cell2zarr convert input.h5ad output.zarr --two-phase --run-db docs/conversion-runs.json
```

### Convert options

| Option | Description |
|--------|-------------|
| `--two-phase` | Two-phase streaming conversion. Low memory usage. |
| `--var-chunk-size INT` | Variable (gene) chunk size. Default: 10. |
| `--cell-chunk-size INT` | Cells per chunk in phase 1. Default: 10000. |
| `--shard-size INT` | Genes per shard (zarr v3 sharding). |
| `--dtype CHOICE` | Output dtype: float16, float32, float64. Default: float32. |
| `--obsm-cell-chunk-size INT` | Cell chunk size for obsm arrays. Default: 50000. |
| `--n-top-genes INT` | Filter to top N highly variable genes. |
| `--keep-raw` | Keep raw counts in the output. |
| `--encoding-config PATH` | Path to JSON encoding config. |
| `--temp-dir PATH` | Directory for phase 1 temp zarr. |
| `--log-file PATH` | Log file path. Default: `<output>.log`. |
| `--log-level CHOICE` | Log level: DEBUG, INFO, WARNING, ERROR. Default: INFO. |
| `--run-db PATH` | Path to JSON run history database. |

### Add options

| Option | Description |
|--------|-------------|
| `--key TEXT` | Key to add (e.g. obsm, obsm/X_umap, obs, X, layers/counts). Required. |
| `--overwrite` | Overwrite existing keys. |
| `--encoding-config PATH` | Path to JSON encoding config. |
| `--dtype CHOICE` | Target dtype. Default: float32. |
| `--log-file PATH` | Log file path. Default: `<zarr_store>.add.log`. |
| `--log-level CHOICE` | Log level: DEBUG, INFO, WARNING, ERROR. Default: INFO. |
| `--temp-dir PATH` | Temp directory for large keys (X, layers). |

## Utility scripts

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
