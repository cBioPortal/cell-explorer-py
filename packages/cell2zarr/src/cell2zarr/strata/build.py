"""Atomic strata table builder."""
from dataclasses import dataclass

import numpy as np
import zarr

from cell2zarr.strata.io import read_obs_categorical


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
