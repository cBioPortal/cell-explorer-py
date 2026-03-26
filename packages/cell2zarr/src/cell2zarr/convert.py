"""h5ad to Zarr conversion pipeline."""
import gc
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import anndata as ad
from anndata.io import write_elem
import numpy as np
from scipy import sparse
import sys
import zarr

from .models import ArrayEncoding, ConversionConfig, EncodingConfig
from .encoding import load_encoding_config, make_compressor, resolve_template


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

    Returns (tmp_root, tmp_dir, phase1_time, has_layers, layer_names).
    """
    has_layers = bool(adata_backed.layers) and config.keep_raw
    layer_names = list(adata_backed.layers.keys()) if has_layers else []
    if not config.keep_raw and adata_backed.layers:
        print(f"Skipping layers (use --keep-raw to include): {list(adata_backed.layers.keys())}", flush=True)

    n_cell_chunks = (n_obs + config.cell_chunk_size - 1) // config.cell_chunk_size

    tmp_dir = tempfile.mkdtemp(prefix="zarr_convert_", dir=config.temp_dir)
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
        compressor_kwarg["compressors"] = make_compressor(x_enc.compressor)

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


def write_obsm_to_store(
    root,
    obsm_data: dict[str, np.ndarray],
    n_obs: int,
    dtype: str = "float32",
    encoding: EncodingConfig | None = None,
    obsm_cell_chunk_size: int = 50000,
) -> None:
    """Write obsm arrays to a zarr store with chunking and sharding."""
    if not obsm_data:
        return

    target_dtype = np.dtype(dtype)
    obsm_enc = encoding.obsm if encoding else ArrayEncoding()
    obsm_compressor_kwarg = {}
    if obsm_enc.compressor:
        obsm_compressor_kwarg["compressors"] = make_compressor(obsm_enc.compressor)

    obsm_group = root.require_group("obsm")
    for key, data in obsm_data.items():
        n_dim = data.shape[1] if data.ndim > 1 else 1
        dim_vars = {"n_dim": n_dim, "n_obs": n_obs}
        obsm_chunks = tuple(resolve_template(v, dim_vars) for v in obsm_enc.chunks) if obsm_enc.chunks else (min(obsm_cell_chunk_size, n_obs), 1)
        obsm_shards = tuple(resolve_template(v, dim_vars) for v in obsm_enc.shards) if obsm_enc.shards else (min(1_000_000, n_obs), n_dim)
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


def _write_metadata(final_root, final_store, metadata: dict, config: ConversionConfig, encoding: EncodingConfig | None = None) -> None:
    """Write obs, var, obsm, uns, obsp, varp to the final zarr store."""
    n_obs = len(metadata["obs"])
    print("Writing metadata...", flush=True)

    # Convert string columns to categoricals for compact storage
    for label, df in [("obs", metadata["obs"]), ("var", metadata["var"])]:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if str_cols:
            print(f"Converting {len(str_cols)} string column(s) in {label} to categorical: {str_cols}", flush=True)
            for c in str_cols:
                df[c] = df[c].astype("category")

    ad.settings.allow_write_nullable_strings = True
    write_elem(final_root, "obs", metadata["obs"])
    write_elem(final_root, "var", metadata["var"])

    obsm_cell_chunk = min(config.obsm_cell_chunk_size, n_obs)
    write_obsm_to_store(final_root, metadata["obsm"], n_obs, config.dtype, encoding, config.obsm_cell_chunk_size)

    # Rechunk obs/_index to align with obsm chunk boundaries
    idx_enc = encoding.obs_index if encoding else ArrayEncoding()
    obs_enc = encoding.obs if encoding else ArrayEncoding()
    idx_compressor_kwarg = {}
    if idx_enc.compressor:
        idx_compressor_kwarg["compressors"] = make_compressor(idx_enc.compressor)
    elif obs_enc.compressor:
        idx_compressor_kwarg["compressors"] = make_compressor(obs_enc.compressor)
    idx_chunk_size = idx_enc.chunks[0] if idx_enc.chunks else obsm_cell_chunk
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


def convert_h5ad_to_zarr_chunked(config: ConversionConfig, hooks: dict[str, Callable] | None = None) -> None:
    """Convert h5ad to dense zarr using two-phase approach.

    Phase 1: Single pass through h5ad → row-chunked temp zarr (aligned with CSR read pattern).
    Phase 2: Rechunk temp zarr → final column-chunked zarr (all_cells, var_chunk_size).

    Optional hooks dict for lifecycle events:
        on_dataset_read(n_obs, n_vars, v_chunk, adata_backed, config, n_cpus)
        on_encoding_loaded(encoding, config_path)
        on_phase1_done(phase1_time)
        on_complete(phase1_time, phase2_time, output_path, config, n_obs, n_vars, v_chunk)
        on_error(exception)
    """
    n_cpus = _init_zarrs()

    print(f"Reading h5ad file in backed mode: {config.input_file}", flush=True)
    adata_backed = ad.read_h5ad(config.input_file, backed="r")
    n_obs, n_vars_orig = adata_backed.shape
    n_vars = n_vars_orig
    print(f"Dataset shape: {n_obs:,} cells x {n_vars:,} genes", flush=True)

    var_idx = _filter_hvgs(adata_backed.var, config.n_top_genes)
    if var_idx is not None:
        n_vars = var_idx.sum()

    v_chunk = min(config.var_chunk_size, n_vars)

    if hooks and "on_dataset_read" in hooks:
        hooks["on_dataset_read"](
            n_obs=n_obs, n_vars=n_vars, v_chunk=v_chunk,
            adata_backed=adata_backed, config=config, n_cpus=n_cpus,
        )

    try:
        # Load encoding config if provided
        encoding = None
        if config.encoding_config:
            variables = {"n_obs": n_obs, "n_vars": n_vars}
            encoding = load_encoding_config(config.encoding_config, variables)
            print(f"Using encoding config: {config.encoding_config}", flush=True)

            if hooks and "on_encoding_loaded" in hooks:
                hooks["on_encoding_loaded"](encoding=encoding, config_path=config.encoding_config)

        tmp_root, tmp_dir, phase1_time, has_layers, layer_names = _phase1_write_temp_zarr(
            config, adata_backed, var_idx, n_obs, n_vars,
        )

        if hooks and "on_phase1_done" in hooks:
            hooks["on_phase1_done"](phase1_time=phase1_time)

        final_root, final_store, phase2_time = _phase2_rechunk(
            config, tmp_root, n_obs, n_vars, v_chunk, has_layers, layer_names, encoding,
        )

        # Clean up temp zarr before loading metadata to free memory
        print(f"Cleaning up temp dir: {tmp_dir}", flush=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        metadata = _extract_metadata(adata_backed, var_idx, n_obs, config.cell_chunk_size)

        _write_metadata(final_root, final_store, metadata, config, encoding)

        total_time = phase1_time + phase2_time
        print(f"\n✓ Done in {total_time:.0f}s (phase1: {phase1_time:.0f}s, phase2: {phase2_time:.0f}s)", flush=True)

        if hooks and "on_complete" in hooks:
            hooks["on_complete"](
                phase1_time=phase1_time, phase2_time=phase2_time,
                output_path=config.output_file, config=config,
                n_obs=n_obs, n_vars=n_vars, v_chunk=v_chunk,
            )

    except Exception as e:
        if hooks and "on_error" in hooks:
            hooks["on_error"](exception=e)
        raise
