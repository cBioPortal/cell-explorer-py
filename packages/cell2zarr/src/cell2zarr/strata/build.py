"""Atomic strata table builder."""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import zarr

from cell2zarr.strata.config import StrataConfig
from cell2zarr.strata.io import read_obs_categorical, read_x_block


@dataclass
class AtomicTable:
    """The four accumulators + their axis labels.

    sum_x  : (n_strata, n_genes) float32 — sum of expression
    sum_xx : (n_strata, n_genes) float32 — sum of squared expression
    nnz    : (n_strata, n_genes) int32   — count of non-zero cells
    n_cells: (n_strata,)         int32   — cell count per stratum
    """
    axis_names: list[str]
    stratum_keys: np.ndarray   # (n_strata, n_axes) string
    sum_x: np.ndarray          # (n_strata, n_genes) float32
    sum_xx: np.ndarray         # (n_strata, n_genes) float32
    nnz: np.ndarray            # (n_strata, n_genes) int32
    n_cells: np.ndarray        # (n_strata,) int32


def compute_strata_mapping(
    root: zarr.Group,
    axis_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify atomic strata from obs categoricals.

    Returns:
        stratum_keys: (n_strata, n_axes) string array — one row per populated stratum
        cell_to_stratum: (n_obs,) int32 — each cell's stratum id
        n_cells: (n_strata,) int32 — cell count per stratum
    """
    columns = [read_obs_categorical(root, name) for name in axis_names]

    # Stack codes; one row per cell, one column per axis.
    code_matrix = np.column_stack([c.codes.astype(np.int64) for c in columns])

    # Find unique rows (i.e., unique stratum tuples). The inverse mapping gives
    # us each cell's stratum id.
    unique_codes, cell_to_stratum_64 = np.unique(
        code_matrix, axis=0, return_inverse=True
    )
    # Guard against numpy versions that return inverse with an extra dimension.
    cell_to_stratum = np.asarray(cell_to_stratum_64).ravel().astype(np.int32)
    n_strata = len(unique_codes)

    # Materialize stratum_keys as a (n_strata, n_axes) string array using each
    # axis's category list.
    stratum_keys = np.empty((n_strata, len(axis_names)), dtype=object)
    for axis_idx, col in enumerate(columns):
        categories = np.asarray(col.categories)
        stratum_keys[:, axis_idx] = categories[unique_codes[:, axis_idx]]
    # Convert object dtype to a fixed-width string dtype for stable zarr storage.
    stratum_keys = stratum_keys.astype("<U32")

    # Cell count per stratum.
    n_cells = np.bincount(cell_to_stratum, minlength=n_strata).astype(np.int32)

    return stratum_keys, cell_to_stratum, n_cells


GENE_BATCH = 30  # genes per accumulator iteration


def build_atomic(root: zarr.Group, config: StrataConfig) -> AtomicTable:
    """Build the atomic strata table by streaming X column-blocks.

    Algorithm:
      1. compute_strata_mapping() → stratum_keys, cell_to_stratum, n_cells
      2. Construct a sparse one-hot indicator (n_obs, n_atomic).
      3. For each gene batch:
           block_f32 = X[:, start:end].astype(float32)  ← upcast BEFORE math
           sum_x  += indicator.T @ block_f32
           sum_xx += indicator.T @ (block_f32 ** 2)
           nnz    += indicator.T @ (block_f32 > 0).astype(float32)
    """
    stratum_keys, cell_to_stratum, n_cells = compute_strata_mapping(
        root, config.atomic_axes
    )
    n_atomic = len(stratum_keys)
    n_genes = int(root["X"].shape[1])

    # Sparse one-hot: indicator[cell_i, stratum] = 1 iff cell_i is in that stratum.
    n_obs = cell_to_stratum.shape[0]
    indicator = sps.csr_matrix(
        (
            np.ones(n_obs, dtype=np.float32),
            (np.arange(n_obs), cell_to_stratum),
        ),
        shape=(n_obs, n_atomic),
    )
    indicator_T = indicator.T  # (n_atomic, n_obs), used in every batch

    sum_x = np.zeros((n_atomic, n_genes), dtype=np.float32)
    sum_xx = np.zeros((n_atomic, n_genes), dtype=np.float32)
    nnz = np.zeros((n_atomic, n_genes), dtype=np.int32)

    for start in range(0, n_genes, GENE_BATCH):
        end = min(start + GENE_BATCH, n_genes)
        block_f32 = read_x_block(root, slice(start, end))  # (n_obs, k), already float32

        # Each multiply is (n_atomic, n_obs) @ (n_obs, k) → (n_atomic, k)
        sum_x[:, start:end] = indicator_T @ block_f32
        sum_xx[:, start:end] = indicator_T @ (block_f32 * block_f32)
        nnz_block = indicator_T @ (block_f32 > 0).astype(np.float32)
        nnz[:, start:end] = nnz_block.astype(np.int32)

    return AtomicTable(
        axis_names=list(config.atomic_axes),
        stratum_keys=stratum_keys,
        sum_x=sum_x,
        sum_xx=sum_xx,
        nnz=nnz,
        n_cells=n_cells,
    )
