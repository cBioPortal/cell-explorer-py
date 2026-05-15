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

## `build-strata` — precompute aggregate tables (optional)

After `convert`, the column-chunked Zarr is already fast for single-gene reads. `build-strata` adds a second precomputed surface for **per-group aggregate queries** (dotplot, cluster DE, marker discovery, pseudobulk DE) — it writes `sum_x`, `sum_xx`, `nnz`, `n_cells` per `(stratum, gene)` into `uns/strata/atomic/` and any requested coarse tables under `uns/strata/coarse_<axes>/`.

**Default: off.** Most everyday conversion / exploration doesn't need strata. Run it when preparing a dataset for production use.

### Examples

Build atomic only:

```bash
uv run cell2zarr build-strata dataset.zarr --atomic-axes cell_type donor condition
```

Atomic + a coarse cell-type table (for browser-side dotplots / cluster DE):

```bash
uv run cell2zarr build-strata dataset.zarr \
  --atomic-axes cell_type donor condition \
  --coarse cell_type \
  --coarse cell_type,treatment
```

YAML config:

```yaml
# strata.yaml
atomic_axes: [cell_type, donor, condition]
coarse:
  - cell_type
  - [cell_type, treatment]
```

```bash
uv run cell2zarr build-strata dataset.zarr --config strata.yaml
```

Rebuild after re-clustering (overwrites existing tables):

```bash
uv run cell2zarr build-strata dataset.zarr --atomic-axes leiden_res10 donor --force
```

## Run dashboard

View conversion run history, configs, performance metrics, and logs in a web dashboard:

```bash
cell2zarr dashboard --run-db /path/to/conversion-runs.json
```

Options:

| Option | Description |
|--------|-------------|
| `--run-db PATH` | Path to JSON run history database. Required. |
| `--host TEXT` | Host to bind to. Default: 127.0.0.1. |
| `--port INT` | Port to bind to. Default: 8000. |
| `--reload` | Enable auto-reload on code changes. |

Open http://localhost:8000. If your run-db is on a remote filesystem (e.g. mounted from a cluster), the dashboard shows real-time status updates as conversions progress.

You can also start the dashboard directly with uvicorn:

```bash
CELL2ZARR_RUN_DB=/path/to/conversion-runs.json uvicorn cell2zarr.dashboard.main:app
```

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
