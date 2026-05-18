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


def strata_variances(table: StrataTable) -> np.ndarray:
    """Per-(stratum, gene) sample variance: (sum_xx - sum_x²/n) / (n - 1).

    Returns (n_strata, n_genes) float32. Strata with n_cells < 2 produce 0.

    Numerical note: the (sum_xx - sum_x²/n) form can suffer cancellation when
    sums are large and variance is small. Acceptable for v1 — strata tables
    hold float32 sums and consumers are visualizing aggregates, not running
    statistical tests.
    """
    out = np.zeros(table.sum_x.shape, dtype=np.float32)
    has_two = table.n_cells >= 2
    if has_two.any():
        n = table.n_cells[has_two].astype(np.float32)[:, None]
        sx = table.sum_x[has_two]
        sxx = table.sum_xx[has_two]
        out[has_two] = (sxx - (sx * sx) / n) / (n - 1)
    return out
