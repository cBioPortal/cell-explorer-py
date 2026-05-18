"""Pure numpy helpers over StrataTable.

These functions take a (Coarse|Atomic)StrataTable and return derived numpy
arrays. No I/O, no class state, no side effects. Strata with n_cells=0
produce 0 in the output (no NaN propagation).
"""

import numpy as np

from zarr_access.strata_store import StrataTable


def strata_means(table: StrataTable) -> np.ndarray:
    """Per-(stratum, gene) mean expression: sum_x / n_cells.

    Returns (n_strata, n_genes) float32. Strata with n_cells=0 produce 0.
    """
    out = np.zeros_like(table.sum_x, dtype=np.float32)
    nonzero = table.n_cells > 0
    if nonzero.any():
        out[nonzero] = table.sum_x[nonzero] / table.n_cells[nonzero, None]
    return out


def strata_frac_expressing(table: StrataTable) -> np.ndarray:
    """Per-(stratum, gene) fraction of cells with nonzero expression: nnz / n_cells.

    Returns (n_strata, n_genes) float32. Strata with n_cells=0 produce 0.
    """
    out = np.zeros(table.nnz.shape, dtype=np.float32)
    nonzero = table.n_cells > 0
    if nonzero.any():
        out[nonzero] = table.nnz[nonzero].astype(np.float32) / table.n_cells[nonzero, None]
    return out
