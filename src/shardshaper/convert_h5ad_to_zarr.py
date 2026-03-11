#!/usr/bin/env python3
"""
Convert h5ad file to zarr format.
"""
import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import anndata as ad
from anndata.io import write_elem
import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy import sparse
import sys
import zarr


class CompressorSpec(BaseModel):
    """Compressor specification for zarr arrays."""
    name: str
    level: int = 0
    cname: str = "lz4"  # blosc only


class ArrayEncoding(BaseModel):
    """Encoding specification for a zarr array."""
    chunks: list[int | str] | None = None
    shards: list[int | str] | None = None
    dtype: str | None = None
    compressor: CompressorSpec | None = None


class EncodingConfig(BaseModel):
    """Full encoding config for all array types."""
    X: ArrayEncoding = Field(default_factory=ArrayEncoding)
    obsm: ArrayEncoding = Field(default_factory=ArrayEncoding)
    obs: ArrayEncoding = Field(default_factory=ArrayEncoding)
    obs_index: ArrayEncoding = Field(default_factory=ArrayEncoding, alias="obs/_index")

    model_config = {"populate_by_name": True}


class ConversionConfig(BaseModel):
    """Configuration for chunked h5ad-to-zarr conversion."""
    input_file: Path
    output_file: Path
    var_chunk_size: int = 10
    n_top_genes: int | None = None
    keep_raw: bool = False
    cell_chunk_size: int = 10000
    shard_size: int | None = None
    dtype: str = "float32"
    obsm_cell_chunk_size: int = 50000
    run_log: Path | None = None
    log_dir: Path | None = None
    encoding_config: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"float16", "float32", "float64"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got '{v}'")
        return v


RunStatus = Literal["running", "phase2", "completed", "failed", "cancelled", "killed"]


class RunDataset(BaseModel):
    """Dataset info for a conversion run."""
    model_config = {"extra": "allow"}
    n_obs: int | None = None
    n_vars: int | None = None
    input_size_gb: float | None = None
    input_format: str | None = None


class RunPerformance(BaseModel):
    """Performance metrics for a conversion run."""
    model_config = {"extra": "allow"}
    start_time: str | None = None
    end_time: str | None = None
    phase1_time_s: int | None = None
    phase2_time_s: int | None = None
    total_time_s: int | None = None
    output_size_gb: float | None = None
    avg_chunk_size_mb: float | None = None
    compression_ratio: float | None = None
    phase2_rate_batches_per_min: float | None = None


class RunZarrConfig(BaseModel):
    """Zarr output config for a conversion run."""
    model_config = {"extra": "allow"}
    format: str | None = None
    chunk_shape: list[int] | None = None
    sharding: list[int] | None = None
    dtype: str | None = None
    compression: str | None = None
    target_encoding: dict[str, Any] | None = None
    actual_encoding: dict[str, Any] | None = None


class RunConversionConfig(BaseModel):
    """Conversion config snapshot for a run."""
    model_config = {"extra": "allow"}
    approach: str | None = None
    cell_chunk_size: int | None = None
    temp_var_chunk: int | None = None
    codec_pipeline: str | None = None
    threads: int | None = None
    skip_layers: bool | None = None
    obsm_keys: list[str] | None = None
    phase2_read_batch: int | None = None


class RunEntry(BaseModel):
    """A single entry in the conversion run log."""
    model_config = {"extra": "allow"}
    run: int
    date: str | None = None
    status: RunStatus = "running"
    script_args: dict[str, Any] | None = None
    zarr_config: RunZarrConfig | None = None
    conversion_config: RunConversionConfig | None = None
    dataset: RunDataset | None = None
    performance: RunPerformance | None = None
    notes: str = ""
    log_file: str | None = None


class _LogTee:
    """Tee stdout to both terminal and a log file."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def _read_runs(log_path: Path) -> list[RunEntry]:
    """Read JSON array from run log file."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return []
    with open(log_path) as f:
        return [RunEntry.model_validate(r) for r in json.load(f)]


def _write_runs(log_path: Path, runs: list[RunEntry]) -> None:
    """Write JSON array to run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump([r.model_dump(exclude_none=True) for r in runs], f, indent=2)
        f.write("\n")


def _get_next_run_number(runs: list[RunEntry]) -> int:
    """Get the next run number."""
    if not runs:
        return 1
    return max(r.run for r in runs) + 1


def _resolve_template(value, variables: dict[str, int]):
    """Resolve template strings like '{n_obs}' in encoding config values.

    Returns the original string if the variable is not yet available.
    """
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        key = value[1:-1]
        if key in variables:
            return variables[key]
        return value
    return value


def _load_encoding_config(config_path: Path, variables: dict[str, int]) -> EncodingConfig:
    """Load an encoding config JSON file and resolve template variables.

    Template variables like {n_obs}, {n_vars}, {n_dim} are resolved using
    the provided variables dict. Unresolvable templates are kept as strings.
    """
    with open(config_path) as f:
        raw = json.load(f)

    # Resolve templates in the raw dict before pydantic parsing
    resolved = {}
    for section, encoding in raw.items():
        resolved[section] = {}
        for key, val in encoding.items():
            if isinstance(val, list):
                resolved[section][key] = [_resolve_template(v, variables) for v in val]
            elif isinstance(val, dict):
                resolved[section][key] = val
            else:
                resolved[section][key] = _resolve_template(val, variables)
    return EncodingConfig.model_validate(resolved)


def _make_compressor(spec: CompressorSpec | dict | None):
    """Create a zarr codec from a CompressorSpec or dict."""
    if spec is None:
        return "auto"
    import zarr.codecs
    if isinstance(spec, dict):
        spec = CompressorSpec(**spec)
    name = spec.name.lower()
    if name == "zstd":
        return zarr.codecs.ZstdCodec(level=spec.level)
    elif name == "blosc":
        return zarr.codecs.BloscCodec(cname=spec.cname, clevel=spec.level)
    elif name == "gzip":
        return zarr.codecs.GzipCodec(level=spec.level)
    else:
        raise ValueError(f"Unknown compressor: {name}")


def _compute_output_stats(output_path: Path, n_obs: int, n_vars: int, dtype: str) -> dict:
    """Compute output zarr stats: size, shard count, compression ratio."""
    # Total output size
    total_bytes = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    )
    output_size_gb = round(total_bytes / (1024**3), 1)

    # X subtree stats
    x_path = output_path / "X"
    x_files = [f for f in x_path.rglob("*") if f.is_file() and f.name != "zarr.json"]
    x_bytes = sum(f.stat().st_size for f in x_files)
    n_shard_files = len(x_files)
    avg_shard_size_mb = round(x_bytes / n_shard_files / (1024**2), 1) if n_shard_files else 0

    # Compression ratio: uncompressed size / on-disk X size
    element_size = np.dtype(dtype).itemsize
    uncompressed_bytes = n_obs * n_vars * element_size
    compression_ratio = round(uncompressed_bytes / x_bytes, 1) if x_bytes else 0

    return {
        "output_size_gb": output_size_gb,
        "n_shard_files": n_shard_files,
        "avg_shard_size_mb": avg_shard_size_mb,
        "compression_ratio": compression_ratio,
    }


def _collect_target_encoding(encoding: "EncodingConfig") -> dict[str, Any]:
    """Serialize the resolved encoding config for the run log."""
    result = {}
    for key, section in [("X", encoding.X), ("obsm", encoding.obsm),
                         ("obs", encoding.obs), ("obs/_index", encoding.obs_index)]:
        d = section.model_dump(exclude_none=True)
        if d:
            # Serialize CompressorSpec to plain dict
            if "compressor" in d and isinstance(d["compressor"], dict):
                d["compressor"] = {k: v for k, v in d["compressor"].items()
                                   if k == "name" or (k == "level" and v != 0)}
            result[key] = d
    return result


def _collect_actual_encoding(output_path: Path) -> dict[str, Any]:
    """Read back actual zarr metadata for each array in the output store."""
    store = zarr.open(str(output_path), mode="r")
    result = {}

    def _dir_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    def _array_info(arr, arr_path: Path) -> dict[str, Any]:
        info: dict[str, Any] = {
            "shape": list(arr.shape),
            "chunks": list(arr.chunks),
            "dtype": str(arr.dtype),
        }
        if arr.shards is not None:
            info["shards"] = list(arr.shards)
        size_bytes = _dir_size(arr_path)
        if size_bytes >= 1e9:
            info["size_gb"] = round(size_bytes / 1e9, 2)
        else:
            info["size_mb"] = round(size_bytes / 1e6, 1)
        return info

    # X
    if "X" in store:
        result["X"] = _array_info(store["X"], output_path / "X")

    # obsm arrays
    if "obsm" in store:
        for key in sorted(store["obsm"].keys()):
            try:
                arr = store[f"obsm/{key}"]
                if hasattr(arr, "shape"):
                    result[f"obsm/{key}"] = _array_info(arr, output_path / "obsm" / key)
            except Exception:
                pass

    # obs/_index
    if "obs" in store and "_index" in store["obs"]:
        result["obs/_index"] = _array_info(store["obs/_index"], output_path / "obs" / "_index")

    return result


def convert_h5ad_to_zarr(input_file: Path, output_file: Path, obs_chunk_size: int | None = None, var_chunk_size: int | None = None, n_top_genes: int | None = None, keep_raw: bool = False, sparse_format: str = "csr", force_int32: bool = False, dense: bool = False) -> None:
    """Convert an h5ad file to zarr format."""
    print(f"Reading h5ad file: {input_file}")
    adata = ad.read_h5ad(input_file)

    print(f"Dataset shape: {adata.shape}")

    # Remove raw counts if requested
    if not keep_raw and adata.raw is not None:
        print(f"Removing raw counts to reduce file size")
        adata.raw = None

    # Filter to top highly variable genes if requested
    if n_top_genes:
        if 'highly_variable_rank' in adata.var.columns:
            # Use existing ranking
            top_genes = adata.var.nsmallest(n_top_genes, 'highly_variable_rank').index
            print(f"Filtering to top {len(top_genes)} highly variable genes using 'highly_variable_rank'")
            adata = adata[:, top_genes].copy()
        elif 'vst.variance.standardized' in adata.var.columns:
            # Use vst.variance.standardized for ranking (higher is more variable)
            top_genes = adata.var.nlargest(n_top_genes, 'vst.variance.standardized').index
            print(f"Filtering to top {len(top_genes)} highly variable genes using 'vst.variance.standardized'")
            adata = adata[:, top_genes].copy()
        elif 'highly_variable' in adata.var.columns:
            # Just take the first n_top_genes that are highly variable
            hvg = adata.var[adata.var['highly_variable']].head(n_top_genes).index
            print(f"Filtering to top {len(hvg)} highly variable genes using 'highly_variable' column")
            adata = adata[:, hvg].copy()
        else:
            print(f"Warning: No highly variable gene information found, computing with scanpy")
            import scanpy as sc
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            adata = adata[:, adata.var['highly_variable']].copy()
        # Ensure X matrix is properly copied (not a view)
        if sparse.issparse(adata.X):
            adata.X = sparse.csr_matrix(adata.X)
            print(f"Reconstructed sparse matrix: nnz={adata.X.nnz:,}")
        print(f"New dataset shape: {adata.shape}")

    # Convert sparse matrix format if needed
    if sparse_format == "csc":
        if sparse.issparse(adata.X):
            if not isinstance(adata.X, sparse.csc_matrix):
                print(f"Converting X matrix to CSC format")
                adata.X = adata.X.tocsc()
            else:
                print(f"X matrix is already in CSC format")
        else:
            print(f"Converting dense X matrix to CSC sparse format")
            adata.X = sparse.csc_matrix(adata.X)
    elif sparse_format == "csr":
        if sparse.issparse(adata.X):
            if not isinstance(adata.X, sparse.csr_matrix):
                print(f"Converting X matrix to CSR format")
                adata.X = adata.X.tocsr()
            else:
                print(f"X matrix is already in CSR format")
        else:
            print(f"Converting dense X matrix to CSR sparse format")
            adata.X = sparse.csr_matrix(adata.X)

    # Convert to dense if requested
    if dense and sparse.issparse(adata.X):
        print(f"Converting sparse matrix to dense array")
        adata.X = adata.X.toarray()
        print(f"Dense matrix shape: {adata.X.shape}, dtype: {adata.X.dtype}")

    # Cast sparse matrix indices to int32 if requested (for JavaScript compatibility)
    if force_int32 and sparse.issparse(adata.X):
        import numpy as np
        max_int32 = np.iinfo(np.int32).max
        if adata.X.indptr[-1] > max_int32:
            print(f"ERROR: indptr max value ({adata.X.indptr[-1]:,}) exceeds int32 max ({max_int32:,})")
            print(f"Cannot safely cast to int32. Try reducing the dataset size.")
            sys.exit(1)
        if adata.X.indices.max() > max_int32:
            print(f"ERROR: indices max value ({adata.X.indices.max():,}) exceeds int32 max ({max_int32:,})")
            sys.exit(1)
        print(f"Casting sparse matrix indices/indptr to int32 (nnz: {adata.X.nnz:,})")
        adata.X.indices = adata.X.indices.astype(np.int32)
        adata.X.indptr = adata.X.indptr.astype(np.int32)

    if obs_chunk_size or var_chunk_size:
        obs_chunk = obs_chunk_size if obs_chunk_size else adata.shape[0]
        var_chunk = var_chunk_size if var_chunk_size else adata.shape[1]
        chunks = (obs_chunk, var_chunk)
        print(f"Writing to zarr with chunks {chunks}: {output_file}")
        adata.write_zarr(output_file, chunks=chunks)
    else:
        print(f"Writing to zarr: {output_file}")
        adata.write_zarr(output_file)
    print(f"✓ Successfully converted to zarr format")


def _init_zarrs() -> int:
    """Enable zarrs-python Rust codec pipeline for parallel chunk encoding/decoding.

    Returns the number of CPU threads available.
    """
    import zarrs  # noqa: F401
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    n_cpus = __import__("os").cpu_count() or 8
    print(f"Using zarrs Rust codec pipeline ({n_cpus} threads)", flush=True)
    return n_cpus


def _filter_hvgs(var_df, n_top_genes: int | None):
    """Filter to top highly variable genes.

    Returns a boolean index into var_df, or None if no filtering is needed.
    """
    if not n_top_genes:
        return None

    if "highly_variable_rank" in var_df.columns:
        top_genes = var_df.nsmallest(n_top_genes, "highly_variable_rank").index
        print(f"Filtering to top {len(top_genes)} HVGs using 'highly_variable_rank'")
    elif "vst.variance.standardized" in var_df.columns:
        top_genes = var_df.nlargest(n_top_genes, "vst.variance.standardized").index
        print(f"Filtering to top {len(top_genes)} HVGs using 'vst.variance.standardized'")
    elif "highly_variable" in var_df.columns:
        top_genes = var_df[var_df["highly_variable"]].head(n_top_genes).index
        print(f"Filtering to top {len(top_genes)} HVGs using 'highly_variable'")
    else:
        print("Warning: No HVG info found. Cannot filter in chunked mode.")
        return None

    var_idx = var_df.index.isin(top_genes)
    n_vars = var_idx.sum()
    print(f"Filtered to {n_vars:,} genes")
    return var_idx


def _phase1_write_temp_zarr(config: ConversionConfig, adata_backed, var_idx, n_obs: int, n_vars: int):
    """Phase 1: Single pass through h5ad → row-chunked temp zarr.

    Returns (tmp_root, tmp_dir, phase1_time).
    """
    import tempfile
    import time

    has_layers = bool(adata_backed.layers) and config.keep_raw
    layer_names = list(adata_backed.layers.keys()) if has_layers else []
    if not config.keep_raw and adata_backed.layers:
        print(f"Skipping layers (use --keep-raw to include): {list(adata_backed.layers.keys())}", flush=True)

    n_cell_chunks = (n_obs + config.cell_chunk_size - 1) // config.cell_chunk_size

    tmp_dir = tempfile.mkdtemp(prefix="zarr_convert_")
    tmp_zarr_path = Path(tmp_dir) / "temp.zarr"
    print(f"\n=== Phase 1: Writing row-chunked temp zarr ===", flush=True)
    print(f"Temp location: {tmp_zarr_path}", flush=True)

    tmp_store = zarr.storage.LocalStore(str(tmp_zarr_path))
    tmp_root = zarr.open_group(tmp_store, mode="w", zarr_format=3)

    # Intermediate chunks: (cell_chunk_size, tmp_var_chunk) — aligned with cell reads
    # but split along genes so Phase 2 column reads don't decompress the full row.
    # 1000 genes per chunk → Phase 2 reads a 1000-gene slab (~40 MB) instead of
    # the full 28,476-gene row (~300 MB) per row-chunk. ~7.5x less decompression waste.
    tmp_var_chunk = min(1000, n_vars)
    print(f"Temp zarr chunk shape: ({config.cell_chunk_size}, {tmp_var_chunk})", flush=True)

    tmp_X = tmp_root.create_array(
        "X",
        shape=(n_obs, n_vars),
        chunks=(config.cell_chunk_size, tmp_var_chunk),
        dtype="float32",
        overwrite=True,
    )

    tmp_layers = {}
    if has_layers:
        tmp_layers_group = tmp_root.require_group("layers")
        for ln in layer_names:
            tmp_layers[ln] = tmp_layers_group.create_array(
                ln,
                shape=(n_obs, n_vars),
                chunks=(config.cell_chunk_size, tmp_var_chunk),
                dtype="float32",
                overwrite=True,
            )

    print(f"Streaming {n_obs:,} cells in {n_cell_chunks} chunks of {config.cell_chunk_size:,}", flush=True)
    phase1_start = time.time()

    for ci in range(n_cell_chunks):
        c_start = ci * config.cell_chunk_size
        c_end = min((ci + 1) * config.cell_chunk_size, n_obs)

        if var_idx is not None:
            chunk = adata_backed[c_start:c_end, var_idx].to_memory()
        else:
            chunk = adata_backed[c_start:c_end, :].to_memory()

        X_dense = chunk.X.toarray() if sparse.issparse(chunk.X) else np.asarray(chunk.X)
        tmp_X[c_start:c_end, :] = X_dense.astype(np.float32)

        for ln in layer_names:
            ld = chunk.layers[ln]
            ld_dense = ld.toarray() if sparse.issparse(ld) else np.asarray(ld)
            tmp_layers[ln][c_start:c_end, :] = ld_dense.astype(np.float32)

        if (ci + 1) % 50 == 0 or ci == n_cell_chunks - 1:
            elapsed = time.time() - phase1_start
            pct = (ci + 1) / n_cell_chunks * 100
            print(f"  Cell chunk {ci + 1}/{n_cell_chunks} ({pct:.0f}%) — {elapsed:.0f}s elapsed", flush=True)

        del chunk, X_dense
        gc.collect()

    phase1_time = time.time() - phase1_start
    print(f"Phase 1 complete in {phase1_time:.0f}s", flush=True)

    return tmp_root, tmp_dir, phase1_time, has_layers, layer_names


def _read_obsm_chunked(adata_backed, n_obs: int, cell_chunk_size: int) -> dict[str, np.ndarray]:
    """Read obsm data from backed adata in chunks.

    Returns dict mapping obsm keys to concatenated arrays.
    """
    obsm_data = {}
    if not adata_backed.obsm:
        return obsm_data

    n_cell_chunks = (n_obs + cell_chunk_size - 1) // cell_chunk_size
    for key in adata_backed.obsm.keys():
        print(f"Reading obsm/{key} in chunks...", flush=True)
        parts = []
        for ci in range(n_cell_chunks):
            c_start = ci * cell_chunk_size
            c_end = min((ci + 1) * cell_chunk_size, n_obs)
            chunk = adata_backed[c_start:c_end, :].to_memory()
            parts.append(np.array(chunk.obsm[key]))
            del chunk
        obsm_data[key] = np.concatenate(parts, axis=0)
        del parts
        gc.collect()

    return obsm_data


def _extract_metadata(adata_backed, var_idx, n_obs: int, cell_chunk_size: int) -> dict:
    """Collect all metadata from backed adata and close the file.

    Returns a dict with keys: obs, var, obsm, uns, obsp, varp.
    """
    adata_file = adata_backed.file
    obs_df = adata_backed.obs.copy()
    var_df_out = adata_backed.var[var_idx].copy() if var_idx is not None else adata_backed.var.copy()

    obsm_data = _read_obsm_chunked(adata_backed, n_obs, cell_chunk_size)

    uns_data = dict(adata_backed.uns) if adata_backed.uns else None
    obsp_data = dict(adata_backed.obsp) if adata_backed.obsp else None
    varp_data = dict(adata_backed.varp) if adata_backed.varp else None

    if adata_file is not None:
        adata_file.close()
    del adata_backed
    gc.collect()

    return {
        "obs": obs_df,
        "var": var_df_out,
        "obsm": obsm_data,
        "uns": uns_data,
        "obsp": obsp_data,
        "varp": varp_data,
    }


def _phase2_rechunk(config: ConversionConfig, tmp_root, n_obs: int, n_vars: int, v_chunk: int, has_layers: bool, layer_names: list[str], encoding: EncodingConfig | None = None):
    """Phase 2: Rechunk temp zarr → final column-chunked zarr.

    Returns (final_root, final_store, phase2_time).
    """
    import time

    target_dtype = np.dtype(config.dtype)

    shard_msg = ""
    shard_kwarg = {}
    compressor_kwarg = {}
    if config.shard_size is not None:
        shard_kwarg["shards"] = (n_obs, config.shard_size)
        shard_msg = f", shards=({n_obs:,}, {config.shard_size})"

    # Encoding config compressor for X
    x_enc = encoding.X if encoding else ArrayEncoding()
    if x_enc.compressor:
        compressor_kwarg["compressors"] = _make_compressor(x_enc.compressor)

    print(f"\n=== Phase 2: Rechunking to ({n_obs:,}, {v_chunk}){shard_msg}, dtype={target_dtype} ===", flush=True)

    final_store = zarr.storage.LocalStore(str(config.output_file))
    final_root = zarr.open_group(final_store, mode="w", zarr_format=3)
    final_root.attrs["encoding-type"] = "anndata"
    final_root.attrs["encoding-version"] = "0.1.0"

    final_X = final_root.create_array(
        "X",
        shape=(n_obs, n_vars),
        chunks=(n_obs, v_chunk),
        dtype=target_dtype,
        overwrite=True,
        **shard_kwarg,
        **compressor_kwarg,
    )
    final_X.attrs["encoding-type"] = "array"
    final_X.attrs["encoding-version"] = "0.2.0"

    final_layers = {}
    if has_layers:
        final_layers_group = final_root.require_group("layers")
        for ln in layer_names:
            fl = final_layers_group.create_array(
                ln,
                shape=(n_obs, n_vars),
                chunks=(n_obs, v_chunk),
                dtype=target_dtype,
                overwrite=True,
                **shard_kwarg,
                **compressor_kwarg,
            )
            fl.attrs["encoding-type"] = "array"
            fl.attrs["encoding-version"] = "0.2.0"
            final_layers[ln] = fl

    # Batch reads by shard size (or var_chunk_size if no sharding) to amortize
    # temp zarr decompression cost. Reading 30 genes at once costs the same
    # decompression as reading 1, but does 30x fewer iterations.
    read_batch = config.shard_size if config.shard_size is not None else v_chunk
    n_batches = (n_vars + read_batch - 1) // read_batch
    print(f"Reading column slices from temp zarr in batches of {read_batch} ({n_batches} batches)...", flush=True)
    phase2_start = time.time()

    tmp_X = tmp_root["X"]
    tmp_layers_data = {ln: tmp_root[f"layers/{ln}"] for ln in layer_names} if has_layers else {}

    for bi in range(n_batches):
        b_start = bi * read_batch
        b_end = min((bi + 1) * read_batch, n_vars)

        # Read batch of columns from temp zarr (amortizes decompression across batch)
        col_data = np.array(tmp_X[:, b_start:b_end]).astype(target_dtype)
        final_X[:, b_start:b_end] = col_data

        for ln in layer_names:
            layer_col = np.array(tmp_layers_data[ln][:, b_start:b_end]).astype(target_dtype)
            final_layers[ln][:, b_start:b_end] = layer_col

        if (bi + 1) % 50 == 0 or bi == n_batches - 1:
            elapsed = time.time() - phase2_start
            pct = (bi + 1) / n_batches * 100
            print(f"  Batch {bi + 1}/{n_batches} ({pct:.0f}%) — {elapsed:.0f}s elapsed", flush=True)

    phase2_time = time.time() - phase2_start
    print(f"Phase 2 complete in {phase2_time:.0f}s", flush=True)

    return final_root, final_store, phase2_time


def _write_metadata(final_root, final_store, metadata: dict, config: ConversionConfig, encoding: EncodingConfig | None = None) -> None:
    """Write obs, var, obsm, uns, obsp, varp to the final zarr store."""
    n_obs = len(metadata["obs"])
    target_dtype = np.dtype(config.dtype)
    print("Writing metadata...", flush=True)

    # Convert string columns to categoricals for compact storage
    for label, df in [("obs", metadata["obs"]), ("var", metadata["var"])]:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if str_cols:
            print(f"Converting {len(str_cols)} string column(s) in {label} to categorical: {str_cols}", flush=True)
            for c in str_cols:
                df[c] = df[c].astype("category")

    write_elem(final_root, "obs", metadata["obs"])
    write_elem(final_root, "var", metadata["var"])

    obsm_data = metadata["obsm"]
    cell_chunk = min(config.obsm_cell_chunk_size, n_obs)
    obsm_enc = encoding.obsm if encoding else ArrayEncoding()
    obsm_compressor_kwarg = {}
    if obsm_enc.compressor:
        obsm_compressor_kwarg["compressors"] = _make_compressor(obsm_enc.compressor)
    if obsm_data:
        obsm_group = final_root.require_group("obsm")
        for key, data in obsm_data.items():
            n_dim = data.shape[1] if data.ndim > 1 else 1
            # Resolve {n_dim} in obsm encoding per key
            dim_vars = {"n_dim": n_dim, "n_obs": n_obs}
            obsm_chunks = tuple(_resolve_template(v, dim_vars) for v in obsm_enc.chunks) if obsm_enc.chunks else (min(100_000, n_obs), 1)
            obsm_shards = tuple(_resolve_template(v, dim_vars) for v in obsm_enc.shards) if obsm_enc.shards else (min(1_000_000, n_obs), n_dim)
            # Shard dims must be exact multiples of chunk dims (zarr v3 requirement)
            obsm_shards = tuple(((s + c - 1) // c) * c for s, c in zip(obsm_shards, obsm_chunks))
            print(f"Writing obsm/{key} shape=({n_obs}, {n_dim}), chunks={obsm_chunks}, shards={obsm_shards}, dtype={target_dtype}...", flush=True)
            zarr_embed = obsm_group.create_array(
                key,
                shape=(n_obs, n_dim),
                chunks=obsm_chunks,
                shards=obsm_shards,
                dtype=target_dtype,
                overwrite=True,
                **obsm_compressor_kwarg,
            )
            zarr_embed.attrs["encoding-type"] = "array"
            zarr_embed.attrs["encoding-version"] = "0.2.0"
            zarr_embed[:] = data.astype(target_dtype)

    # Rechunk obs/_index to align with obsm chunk boundaries
    idx_enc = encoding.obs_index if encoding else ArrayEncoding()
    obs_enc = encoding.obs if encoding else ArrayEncoding()
    idx_compressor_kwarg = {}
    if idx_enc.compressor:
        idx_compressor_kwarg["compressors"] = _make_compressor(idx_enc.compressor)
    elif obs_enc.compressor:
        idx_compressor_kwarg["compressors"] = _make_compressor(obs_enc.compressor)
    idx_chunk_size = idx_enc.chunks[0] if idx_enc.chunks else cell_chunk
    idx_shard_kwarg = {}
    if idx_enc.shards:
        idx_shard_kwarg["shards"] = (idx_enc.shards[0],)
    if "obs" in final_root:
        obs_group = final_root["obs"]
        if "_index" in obs_group:
            old_index = obs_group["_index"]
            index_data = old_index[:]
            old_attrs = dict(old_index.attrs)
            del obs_group["_index"]
            shard_msg = f", shards=({idx_enc.shards[0]},)" if idx_enc.shards else ""
            print(f"Rechunking obs/_index to chunks=({idx_chunk_size},){shard_msg} ...", flush=True)
            new_index = obs_group.create_array(
                "_index",
                shape=index_data.shape,
                chunks=(idx_chunk_size,),
                dtype=index_data.dtype,
                overwrite=True,
                **idx_shard_kwarg,
                **idx_compressor_kwarg,
            )
            new_index[:] = index_data
            new_index.attrs.update(old_attrs)

    if metadata["uns"]:
        print("Writing uns...", flush=True)
        write_elem(final_root, "uns", metadata["uns"])
    if metadata["obsp"]:
        write_elem(final_root, "obsp", metadata["obsp"])
    if metadata["varp"]:
        write_elem(final_root, "varp", metadata["varp"])

    zarr.consolidate_metadata(final_store, zarr_format=3)


def _update_run_entry(log_path: Path, run_number: int, updates: dict) -> None:
    """Update fields on an existing run entry and write back to disk."""
    runs = _read_runs(log_path)
    for i, r in enumerate(runs):
        if r.run == run_number:
            d = r.model_dump(exclude_none=True)
            for key, val in updates.items():
                if isinstance(val, dict) and isinstance(d.get(key), dict):
                    d[key].update(val)
                else:
                    d[key] = val
            runs[i] = RunEntry.model_validate(d)
            break
    _write_runs(log_path, runs)


def convert_h5ad_to_zarr_chunked(config: ConversionConfig) -> None:
    """Convert h5ad to dense zarr using two-phase approach.

    Phase 1: Single pass through h5ad → row-chunked temp zarr (aligned with CSR read pattern).
    Phase 2: Rechunk temp zarr → final column-chunked zarr (all_cells, var_chunk_size).
    """
    import shutil

    n_cpus = _init_zarrs()

    # --- Run logging: write initial "running" entry before h5ad read ---
    run_number = None
    log_tee = None
    if config.run_log:
        runs = _read_runs(config.run_log)
        run_number = _get_next_run_number(runs)

        log_dir = config.log_dir or config.run_log.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"run-{run_number}.log"
        log_file_rel = f"logs/run-{run_number}.log"

        input_size_gb = round(config.input_file.stat().st_size / (1024**3), 1)

        script_args = {
            "input": str(config.input_file),
            "output": str(config.output_file),
            "--two-phase": True,
            "--var-chunk-size": config.var_chunk_size,
            "--keep-raw": config.keep_raw,
            "--cell-chunk-size": config.cell_chunk_size,
            "--dtype": config.dtype,
            "--obsm-cell-chunk-size": config.obsm_cell_chunk_size,
        }
        if config.shard_size is not None:
            script_args["--shard-size"] = config.shard_size
        if config.n_top_genes is not None:
            script_args["--n-top-genes"] = config.n_top_genes
        if config.encoding_config is not None:
            script_args["--encoding-config"] = str(config.encoding_config)

        run_entry = RunEntry(
            run=run_number,
            date=datetime.now().strftime("%Y-%m-%d"),
            status="running",
            script_args=script_args,
            dataset=RunDataset(
                input_size_gb=input_size_gb,
                input_format="sparse_csr",
            ),
            performance=RunPerformance(
                start_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
            notes="",
            log_file=log_file_rel,
        )
        runs.append(run_entry)
        _write_runs(config.run_log, runs)

        log_tee = _LogTee(log_file_path)
        sys.stdout = log_tee

    print(f"Reading h5ad file in backed mode: {config.input_file}", flush=True)
    adata_backed = ad.read_h5ad(config.input_file, backed="r")
    n_obs, n_vars_orig = adata_backed.shape
    n_vars = n_vars_orig
    print(f"Dataset shape: {n_obs:,} cells x {n_vars:,} genes", flush=True)

    var_idx = _filter_hvgs(adata_backed.var, config.n_top_genes)
    if var_idx is not None:
        n_vars = var_idx.sum()

    v_chunk = min(config.var_chunk_size, n_vars)

    # --- Run logging: update with dataset info after h5ad read ---
    if config.run_log and run_number is not None:
        obsm_keys = list(adata_backed.obsm.keys()) if adata_backed.obsm else []
        shard_str = [n_obs, config.shard_size] if config.shard_size else None
        _update_run_entry(config.run_log, run_number, {
            "zarr_config": {
                "format": "v3",
                "chunk_shape": [n_obs, v_chunk],
                "sharding": shard_str,
                "dtype": config.dtype,
            },
            "conversion_config": {
                "approach": "two-phase-rechunk",
                "cell_chunk_size": config.cell_chunk_size,
                "temp_var_chunk": min(1000, n_vars),
                "codec_pipeline": "zarrs-python",
                "threads": n_cpus,
                "skip_layers": not config.keep_raw,
                "obsm_keys": obsm_keys,
                "phase2_read_batch": config.shard_size if config.shard_size else v_chunk,
            },
            "dataset": {
                "n_obs": n_obs,
                "n_vars": n_vars,
                "input_size_gb": round(config.input_file.stat().st_size / (1024**3), 1),
                "input_format": "sparse_csr",
            },
        })

    try:
        # Load encoding config if provided
        encoding = None
        if config.encoding_config:
            variables = {"n_obs": n_obs, "n_vars": n_vars}
            encoding = _load_encoding_config(config.encoding_config, variables)
            print(f"Using encoding config: {config.encoding_config}", flush=True)

            # --- Run logging: target encoding + raw config ---
            if config.run_log and run_number is not None:
                encoding_config_raw = json.loads(config.encoding_config.read_text())
                _update_run_entry(config.run_log, run_number, {
                    "zarr_config": {
                        "encoding_config_raw": encoding_config_raw,
                        "target_encoding": _collect_target_encoding(encoding),
                    },
                })

        tmp_root, tmp_dir, phase1_time, has_layers, layer_names = _phase1_write_temp_zarr(
            config, adata_backed, var_idx, n_obs, n_vars,
        )

        # --- Run logging: phase 1 complete ---
        if config.run_log and run_number is not None:
            _update_run_entry(config.run_log, run_number, {
                "status": "phase2",
                "performance": {"phase1_time_s": round(phase1_time)},
            })

        metadata = _extract_metadata(adata_backed, var_idx, n_obs, config.cell_chunk_size)

        final_root, final_store, phase2_time = _phase2_rechunk(
            config, tmp_root, n_obs, n_vars, v_chunk, has_layers, layer_names, encoding,
        )

        _write_metadata(final_root, final_store, metadata, config, encoding)

        # Clean up temp zarr
        print(f"Cleaning up temp dir: {tmp_dir}", flush=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        total_time = phase1_time + phase2_time
        print(f"\n✓ Done in {total_time:.0f}s (phase1: {phase1_time:.0f}s, phase2: {phase2_time:.0f}s)", flush=True)

        # --- Run logging: completed ---
        if config.run_log and run_number is not None:
            output_stats = _compute_output_stats(config.output_file, n_obs, n_vars, config.dtype)
            read_batch = config.shard_size if config.shard_size else v_chunk
            n_batches = (n_vars + read_batch - 1) // read_batch
            phase2_rate = round(n_batches / (phase2_time / 60), 1) if phase2_time > 0 else 0
            actual = _collect_actual_encoding(config.output_file)
            _update_run_entry(config.run_log, run_number, {
                "status": "completed",
                "zarr_config": {"actual_encoding": actual},
                "performance": {
                    "end_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "phase1_time_s": round(phase1_time),
                    "phase2_time_s": round(phase2_time),
                    "total_time_s": round(total_time),
                    "phase2_rate_batches_per_min": phase2_rate,
                    **output_stats,
                },
            })

    except Exception as e:
        # --- Run logging: failed ---
        if config.run_log and run_number is not None:
            _update_run_entry(config.run_log, run_number, {
                "status": "failed",
                "performance": {"end_time": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                "notes": f"Error: {e}",
            })
        raise
    finally:
        if log_tee is not None:
            log_tee.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert h5ad file to zarr format"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input h5ad file"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Output zarr directory (defaults to input filename with .zarr extension)"
    )
    parser.add_argument(
        "--obs-chunk-size",
        type=int,
        help="Observation (cell) chunk size. If not specified, uses full dataset size."
    )
    parser.add_argument(
        "--var-chunk-size",
        type=int,
        help="Variable (gene) chunk size. Default: 10. Overrides encoding config."
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        help="Filter to top N highly variable genes before writing to zarr"
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw counts in the output (default: remove to save space)"
    )
    parser.add_argument(
        "--sparse-format",
        choices=["csr", "csc"],
        default="csr",
        help="Sparse matrix format: csr (row-oriented, good for cell access) or csc (column-oriented, good for gene access). Default: csr"
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Convert sparse matrix to dense array before writing to zarr"
    )
    parser.add_argument(
        "--force-int32",
        action="store_true",
        help="Cast sparse matrix indices/indptr to int32 for JavaScript/Vitessce compatibility. Will fail if values exceed int32 max."
    )
    parser.add_argument(
        "--two-phase",
        action="store_true",
        help="Use two-phase streaming conversion: reads h5ad in backed mode, writes row-chunked temp zarr, then rechunks to final column layout. Low memory usage."
    )
    parser.add_argument(
        "--cell-chunk-size",
        type=int,
        default=10000,
        help="Number of cells per chunk in phase 1 streaming. Default: 10000"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        help="Number of genes per shard (zarr v3 sharding). If not set, no sharding is used."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "float64"],
        help="Data type for the output X matrix and obsm embeddings. Default: float32. Overrides encoding config."
    )
    parser.add_argument(
        "--obsm-cell-chunk-size",
        type=int,
        help="Cell chunk size for obsm embedding arrays (enables progressive loading). Default: 50000. Overrides encoding config."
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        help="Path to JSON run log file (e.g. docs/conversion-runs.json). No logging if omitted."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for stdout log files. Defaults to <run-log-dir>/logs/"
    )
    parser.add_argument(
        "--encoding-config",
        type=Path,
        help="Path to JSON encoding config file specifying chunks, shards, dtype, and compressor for X, obsm, and obs arrays."
    )
    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Set output file if not provided
    if args.output_file is None:
        args.output_file = args.input_file.with_suffix('.zarr')

    # Check if output already exists
    if args.output_file.exists():
        print(f"Error: Output '{args.output_file}' already exists", file=sys.stderr)
        sys.exit(1)

    # Convert
    if args.two_phase:
        if args.sparse_format != "csr" or args.force_int32:
            print("Note: --sparse-format and --force-int32 are ignored in two-phase mode", file=sys.stderr)

        # Apply encoding config defaults for CLI args not explicitly set
        enc_defaults = {}
        if args.encoding_config and args.encoding_config.exists():
            enc = EncodingConfig.model_validate_json(args.encoding_config.read_text())
            if enc.X.chunks and len(enc.X.chunks) > 1 and isinstance(enc.X.chunks[1], int):
                enc_defaults["var_chunk_size"] = enc.X.chunks[1]
            if enc.X.shards and len(enc.X.shards) > 1 and isinstance(enc.X.shards[1], int):
                enc_defaults["shard_size"] = enc.X.shards[1]
            if enc.X.dtype:
                enc_defaults["dtype"] = enc.X.dtype
            if enc.obsm.chunks and len(enc.obsm.chunks) > 0 and isinstance(enc.obsm.chunks[0], int):
                enc_defaults["obsm_cell_chunk_size"] = enc.obsm.chunks[0]

        config = ConversionConfig(
            input_file=args.input_file,
            output_file=args.output_file,
            var_chunk_size=args.var_chunk_size if args.var_chunk_size is not None else enc_defaults.get("var_chunk_size", 10),
            n_top_genes=args.n_top_genes,
            keep_raw=args.keep_raw,
            cell_chunk_size=args.cell_chunk_size,
            shard_size=args.shard_size if args.shard_size is not None else enc_defaults.get("shard_size"),
            dtype=args.dtype if args.dtype is not None else enc_defaults.get("dtype", "float32"),
            obsm_cell_chunk_size=args.obsm_cell_chunk_size if args.obsm_cell_chunk_size is not None else enc_defaults.get("obsm_cell_chunk_size", 50000),
            run_log=args.run_log,
            log_dir=args.log_dir,
            encoding_config=args.encoding_config,
        )
        convert_h5ad_to_zarr_chunked(config)
    else:
        convert_h5ad_to_zarr(args.input_file, args.output_file, args.obs_chunk_size, args.var_chunk_size, args.n_top_genes, args.keep_raw, args.sparse_format, args.force_int32, args.dense)


if __name__ == "__main__":
    main()
