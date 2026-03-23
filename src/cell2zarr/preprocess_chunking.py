# %%
from pathlib import Path
import gc

import click
import h5py
import pandas as pd
import scanpy as sc
from scipy import sparse

from anndata import AnnData
import anndata.io

# %%
OBS_NAMES_KEY = "cell_name"  # obs index key
VAR_NAMES_KEY = "gene_symbol"  # var index key
SAMPLE_KEY = "sample"
CELL_TYPE_KEY = "cell_type"
AUTHOR_CELL_TYPE_KEY = "author_cell_type"
TECHNOLOGY_KEY = "technology"
SITE_KEY = "site"
DISEASE_KEY = "disease"
EXPRESSION_UNIT_KEY = "expression_unit"  # can be either "UMI", "TPM", or "TP10K"
DATASET_KEY = "dataset_id"
DONOR_KEY = "patient"


# %%
def use_counts_layer(adata, verbose=False):
    """
    Use counts layer as X matrix.
    The X matrix contains normalized data, but we need raw UMI counts.
    """
    if "counts" not in adata.layers:
        raise ValueError("No 'counts' layer found! ScopeAtlas requires raw UMI counts.")

    if verbose:
        print("Using 'counts' layer as main expression matrix")
        print("  (X matrix contains normalized data; counts layer has raw UMI counts)")

    adata.X = adata.layers["counts"]

    # Ensure X is in CSR sparse format for optimal performance
    if not sparse.issparse(adata.X):
        if verbose:
            print("  Converting X to CSR sparse matrix...")
        adata.X = sparse.csr_matrix(adata.X)
    elif not isinstance(adata.X, sparse.csr_matrix):
        if verbose:
            print(
                f"  Converting X from {type(adata.X).__name__} to CSR sparse matrix..."
            )
        adata.X = adata.X.tocsr()
    else:
        if verbose:
            print("  X is already in CSR sparse format")

    return adata


# %%
def harmonize_cell_types(adata, verbose=False):
    """Map CRC Atlas cell type columns to standard cell_type column."""
    cell_type_cols = [
        col
        for col in adata.obs.columns
        if ("cell_type" in col.lower()) or ("celltype" in col.lower())
    ]
    if verbose:
        print(f"Cell type columns: {cell_type_cols}")

    # CRC Atlas already has a 'cell_type' column with harmonized cell types
    if CELL_TYPE_KEY not in adata.obs.columns:
        raise ValueError(f"'{CELL_TYPE_KEY}' column not found in obs")

    if verbose:
        print(f"  Using existing '{CELL_TYPE_KEY}' column")
        print(f"  Found {adata.obs[CELL_TYPE_KEY].nunique()} unique cell types")


# %%
def add_required_cols(adata, verbose=False):
    """Map CRC Atlas columns to required ScopeAtlas column names."""
    # Map column names
    column_mapping = {
        "sample_id": SAMPLE_KEY,
        "donor_id": DONOR_KEY,
        "platform": TECHNOLOGY_KEY,
        "tissue": SITE_KEY,
        "cell_type_study": AUTHOR_CELL_TYPE_KEY,
        "dataset": DATASET_KEY,
    }

    for old_col, new_col in column_mapping.items():
        if old_col in adata.obs.columns and new_col not in adata.obs.columns:
            adata.obs[new_col] = adata.obs[old_col].copy()
            if verbose:
                print(f"  Mapped: {old_col} → {new_col}")

    # Set expression_unit to UMI for raw counts
    adata.obs[EXPRESSION_UNIT_KEY] = "UMI"

    # Ensure disease column exists
    if DISEASE_KEY not in adata.obs.columns:
        raise ValueError(f"'{DISEASE_KEY}' column not found in obs")

    if verbose:
        print(f"  Set {EXPRESSION_UNIT_KEY} to 'UMI'")


# %%
@click.command()
@click.option("--adata_path", type=click.Path(exists=True), required=True)
@click.option("--save_path", type=click.Path(), required=True)
@click.option("--dataset_id", type=str, default="CRCATLAS/crc_atlas_2024")
@click.option(
    "--chunk_size",
    type=int,
    default=300000,
    help="Number of cells to process per chunk",
)
@click.option("--verbose", is_flag=True, default=False)
def main(adata_path, save_path, dataset_id, chunk_size, verbose=False):
    adata_path = Path(adata_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading data from {adata_path} in backed mode...")

    # Read in backed mode to avoid loading everything into memory
    adata_backed = sc.read_h5ad(adata_path, backed="r")

    if verbose:
        print(f"Loaded: {adata_backed.n_obs:,} cells x {adata_backed.n_vars:,} genes")

    # Get indices of raw counts cells
    if verbose:
        print("\nFiltering to raw counts cells...")

    if "matrix_type" in adata_backed.obs.columns:
        raw_counts_mask = adata_backed.obs["matrix_type"] == "raw counts"
        raw_counts_indices = raw_counts_mask[raw_counts_mask].index.tolist()
        n_cells = len(raw_counts_indices)

        if verbose:
            print(f"  Found {n_cells:,} raw counts cells")
            print(f"  Excluding {(~raw_counts_mask).sum():,} non-raw-counts cells")
    else:
        raw_counts_indices = adata_backed.obs.index.tolist()
        n_cells = len(raw_counts_indices)
        if verbose:
            print(f"  No 'matrix_type' column found, processing all {n_cells:,} cells")

    # Determine output file
    input_stem = adata_path.stem
    output_file = save_path / f"{input_stem}_preprocessed.h5ad"
    tmp_dir = save_path / "tmp_h5ads"
    tmp_dir.mkdir(exist_ok=True)

    # Process in chunks and create temporary h5ad files
    n_chunks = (n_cells + chunk_size - 1) // chunk_size

    if verbose:
        print(
            f"\nProcessing {n_cells:,} cells in {n_chunks} chunks of up to {chunk_size:,} cells"
        )

    obs_cols_required = [
        SAMPLE_KEY,
        CELL_TYPE_KEY,
        AUTHOR_CELL_TYPE_KEY,
        TECHNOLOGY_KEY,
        EXPRESSION_UNIT_KEY,
        SITE_KEY,
        DISEASE_KEY,
        DONOR_KEY,
        DATASET_KEY,
        OBS_NAMES_KEY,
    ]

    backed_h5ads = []
    obs_dfs = []
    var_df = None

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_cells)
        chunk_indices = raw_counts_indices[start_idx:end_idx]

        if verbose:
            print(f"\nChunk {i + 1}/{n_chunks}: cells {start_idx:,} to {end_idx:,}")

        # Load chunk
        adata_chunk = adata_backed[chunk_indices, :].to_memory()

        # Process chunk
        adata_chunk = use_counts_layer(adata_chunk, verbose=False)
        harmonize_cell_types(adata_chunk, verbose=False)
        add_required_cols(adata_chunk, verbose=False)

        # Add cell names column (original index before prepending dataset_id)
        adata_chunk.obs[OBS_NAMES_KEY] = adata_chunk.obs.index.astype(str)

        # Create processed chunk with only required columns
        # Keep essential gene metadata columns if they exist
        var_cols_to_keep = []
        for col in ["var_names", "GeneSymbol", "ensembl", "Chromosome"]:
            if col in adata_chunk.var.columns:
                var_cols_to_keep.append(col)

        if var_cols_to_keep:
            var_data = adata_chunk.var[var_cols_to_keep].copy()
        else:
            var_data = pd.DataFrame(index=adata_chunk.var.index)

        # Add gene symbol column to var (use GeneSymbol if available, else var_names)
        if "GeneSymbol" in adata_chunk.var.columns:
            var_data[VAR_NAMES_KEY] = adata_chunk.var["GeneSymbol"]
        elif "var_names" in adata_chunk.var.columns:
            var_data[VAR_NAMES_KEY] = adata_chunk.var["var_names"]
        else:
            # Fallback to using the index as gene symbols
            var_data[VAR_NAMES_KEY] = adata_chunk.var.index.astype(str)

        adata_processed = AnnData(
            X=adata_chunk.X.copy(),
            obs=adata_chunk.obs[obs_cols_required].copy(),
            var=var_data,
        )

        # Prepend dataset ID to cell names (modifies the index)
        adata_processed.obs.index = (
            dataset_id + "_" + adata_processed.obs.index.astype(str)
        )

        # Save var_df from first chunk
        if var_df is None:
            var_df = adata_processed.var.copy()

        # Collect obs for later
        obs_dfs.append(adata_processed.obs.copy())

        # Write chunk to temporary file
        chunk_path = tmp_dir / f"chunk_{i}.h5ad"
        adata_processed.write(chunk_path)

        # Read back in backed mode to save memory
        backed_h5ads.append(sc.read_h5ad(chunk_path, backed="r"))

        del adata_chunk, adata_processed
        gc.collect()

        if verbose:
            print(f"  Processed and saved chunk {i + 1}")

    # Concatenate obs dataframes
    if verbose:
        print(f"\nConcatenating metadata...")
    obs_df = pd.concat(obs_dfs, ignore_index=False)

    # Convert all obs columns to strings and handle missing values
    # This is required for HDF5 string storage
    for col in obs_df.columns:
        # Convert to string first (handles Categorical), then fill missing
        obs_df[col] = (
            obs_df[col].astype(str).replace("nan", "unknown").replace("None", "unknown")
        )

    # Convert all var columns to strings and handle missing values
    for col in var_df.columns:
        # Convert to string first (handles Categorical), then fill missing
        var_df[col] = (
            var_df[col].astype(str).replace("nan", "unknown").replace("None", "unknown")
        )

    # Concatenate expression matrices on disk using h5py (similar to ScopeAtlas.export_to_h5ad)
    if verbose:
        print(f"\nWriting to {output_file}...")
        print(f"  Concatenating {n_chunks} chunks on disk...")

    with h5py.File(output_file, "w") as f:
        # Create empty CSR matrix for X
        import numpy as np

        csr_empty = sparse.csr_matrix(
            (0, backed_h5ads[0].X.shape[1]), dtype=backed_h5ads[0].X.dtype
        )
        csr_empty.indices, csr_empty.indptr = sparse.safely_cast_index_arrays(
            csr_empty, np.int64
        )
        anndata.io.write_elem(f, "X", csr_empty)

        # Add gzip compression to the sparse matrix datasets
        if verbose:
            print(f"  Enabling gzip compression on sparse matrix datasets...")

        # We need to recreate the datasets with compression
        # Store the shape and dtype info first
        n_genes = backed_h5ads[0].X.shape[1]

        # Remove the uncompressed datasets
        del f["X"]["data"]
        del f["X"]["indices"]
        del f["X"]["indptr"]

        # Create compressed datasets with chunking for efficient appending
        # Start with initial size estimates
        f["X"].create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            compression="gzip",
            chunks=True,
        )
        f["X"].create_dataset(
            "indices",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            compression="gzip",
            chunks=True,
        )
        f["X"].create_dataset(
            "indptr",
            shape=(1,),
            maxshape=(None,),
            dtype=np.int64,
            compression="gzip",
            chunks=True,
        )

        # Initialize indptr with 0
        f["X"]["indptr"][0] = 0

        # Manually append each chunk's sparse matrix data
        for i, backed_adata in enumerate(backed_h5ads):
            if verbose and (i + 1) % 5 == 0:
                print(f"  Appending chunk {i + 1}/{n_chunks}...")

            # Convert backed chunk to memory to access sparse matrix components
            # These temporary chunks are small enough to fit in memory
            chunk_adata = backed_adata.to_memory()
            chunk_X = chunk_adata.X

            if not isinstance(chunk_X, sparse.csr_matrix):
                chunk_X = chunk_X.tocsr()

            chunk_data = chunk_X.data
            chunk_indices = chunk_X.indices
            chunk_indptr = chunk_X.indptr

            # Append data
            data_dset = f["X"]["data"]
            old_data_size = data_dset.shape[0]
            new_data_size = old_data_size + chunk_data.shape[0]
            data_dset.resize((new_data_size,))
            data_dset[old_data_size:new_data_size] = chunk_data

            # Append indices
            indices_dset = f["X"]["indices"]
            indices_dset.resize((new_data_size,))
            indices_dset[old_data_size:new_data_size] = chunk_indices

            # Append indptr (skip first element which is always 0)
            indptr_dset = f["X"]["indptr"]
            old_indptr_size = indptr_dset.shape[0]
            new_indptr_size = old_indptr_size + chunk_indptr.shape[0] - 1
            indptr_dset.resize((new_indptr_size,))
            # Add offset from previous chunks and skip the first 0
            offset = indptr_dset[old_indptr_size - 1]
            indptr_dset[old_indptr_size:new_indptr_size] = chunk_indptr[1:] + offset

        # Add shape attribute for CSR format
        f["X"].attrs["shape"] = (n_cells, n_genes)
        f["X"].attrs["encoding-type"] = "csr_matrix"
        f["X"].attrs["encoding-version"] = "0.1.0"

        # Write obs and var
        anndata.io.write_elem(f, "obs", obs_df)
        anndata.io.write_elem(f, "var", var_df)

    # Clean up temporary directory
    if verbose:
        print(f"\nCleaning up temporary files...")
    for chunk_path in tmp_dir.glob("*.h5ad"):
        chunk_path.unlink()
    if not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    if verbose:
        import os

        file_size_gb = os.path.getsize(output_file) / (1024**3)
        print(f"\n✓ Preprocessing complete!")
        print(f"  Output: {output_file}")
        print(f"  File size: {file_size_gb:.2f} GB")
        print(f"  Total cells: {n_cells:,}")
        print(f"  Total genes: {var_df.shape[0]:,}")


if __name__ == "__main__":
    main()
