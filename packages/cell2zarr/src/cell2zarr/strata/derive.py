"""Coarse-table derivation by axis-collapse from an in-memory AtomicTable."""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps

from cell2zarr.strata.build import AtomicTable


@dataclass
class CoarseTable:
    """The same four accumulators, on a subset of the atomic axes.

    sum_x  : (n_coarse, n_genes) float32
    sum_xx : (n_coarse, n_genes) float32
    nnz    : (n_coarse, n_genes) int32
    n_cells: (n_coarse,)         int32
    """
    axis_names: list[str]
    stratum_keys: np.ndarray
    sum_x: np.ndarray
    sum_xx: np.ndarray
    nnz: np.ndarray
    n_cells: np.ndarray


def _unique_rows_sorted(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_rows_sorted, inverse) where inverse[i] is unique-index of rows[i].

    Unique rows are lex-sorted for deterministic output.
    """
    if rows.shape[1] == 0:
        # Special case: 0-axis (full collapse) → all rows map to the single
        # "global" stratum with empty key.
        inverse = np.zeros(rows.shape[0], dtype=np.int64)
        return np.zeros((1, 0), dtype=rows.dtype), inverse

    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    # Guard against numpy versions that return inverse with an extra dimension.
    inverse = np.asarray(inverse).ravel()
    # np.unique already sorts lexicographically — confirmed by docs.
    return unique, inverse


def derive_coarse(atomic: AtomicTable, coarse_axes: list[str]) -> CoarseTable:
    """Sum atomic rows along the axes NOT in coarse_axes.

    Empty coarse_axes → single global row (sums across all atomic).
    coarse_axes == atomic.axis_names → coarse is atomic re-sorted lexicographically.
    """
    # Indices of the axes we keep, in coarse_axes order.
    keep_idx = [atomic.axis_names.index(a) for a in coarse_axes]
    keep_cols = atomic.stratum_keys[:, keep_idx] if keep_idx else np.zeros(
        (len(atomic.stratum_keys), 0), dtype=atomic.stratum_keys.dtype
    )

    coarse_keys, atomic_to_coarse = _unique_rows_sorted(keep_cols)
    n_coarse = len(coarse_keys)
    n_atomic = len(atomic.stratum_keys)

    # Aggregator sparse matrix: M[c, a] = 1 iff atomic stratum `a` maps to coarse stratum `c`.
    M = sps.csr_matrix(
        (
            np.ones(n_atomic, dtype=np.float32),
            (atomic_to_coarse, np.arange(n_atomic)),
        ),
        shape=(n_coarse, n_atomic),
    )

    sum_x = (M @ atomic.sum_x).astype(np.float32)
    sum_xx = (M @ atomic.sum_xx).astype(np.float32)
    nnz = (M @ atomic.nnz.astype(np.float32)).astype(np.int32)
    n_cells = (M @ atomic.n_cells.astype(np.float32)).astype(np.int32)

    return CoarseTable(
        axis_names=list(coarse_axes),
        stratum_keys=coarse_keys,
        sum_x=sum_x,
        sum_xx=sum_xx,
        nnz=nnz,
        n_cells=n_cells,
    )
