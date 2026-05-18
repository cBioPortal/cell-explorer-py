"""Shared per-gene compute helper for compare_groups.

Takes per-group sums (sum_x, sum_xx) and cell counts, returns means,
variances, Cohen's d (pooled), and Welch's t-statistic — all as numpy
arrays of shape (n_genes,). Used by both the strata fast path and the
X-scan fallback so the math is byte-identical regardless of routing.
"""

from __future__ import annotations

import numpy as np


def compute_stats(
    sum_x_a: np.ndarray,
    sum_xx_a: np.ndarray,
    n_a: int,
    sum_x_b: np.ndarray,
    sum_xx_b: np.ndarray,
    n_b: int,
) -> dict[str, np.ndarray]:
    """Compute per-gene stats from group sums.

    All input arrays must have shape (n_genes,). Returns a dict with keys
    ``mean_a``, ``mean_b``, ``var_a``, ``var_b``, ``cohens_d``,
    ``t_statistic`` — each a float64 array of shape (n_genes,).

    Cohen's d uses pooled variance:
        d = (mean_a - mean_b) / sqrt(((n_a-1)*var_a + (n_b-1)*var_b) / (n_a+n_b-2))

    Welch's t-stat (no equal-variance assumption):
        t = (mean_a - mean_b) / sqrt(var_a/n_a + var_b/n_b)

    Non-finite outputs (NaN/±Inf) are expected for degenerate gene/group
    combinations (zero variance, equal means) and must be handled by the
    caller's sort step.
    """
    # float64 throughout to mitigate catastrophic cancellation in
    # var = (sum_xx - sum_x**2 / n) / (n-1) when means are large.
    sum_x_a = sum_x_a.astype("float64", copy=False)
    sum_xx_a = sum_xx_a.astype("float64", copy=False)
    sum_x_b = sum_x_b.astype("float64", copy=False)
    sum_xx_b = sum_xx_b.astype("float64", copy=False)

    mean_a = sum_x_a / n_a
    mean_b = sum_x_b / n_b

    # Sample variance. Clip tiny negative residuals (float noise) to 0.
    var_a = np.maximum((sum_xx_a - sum_x_a**2 / n_a) / (n_a - 1), 0.0)
    var_b = np.maximum((sum_xx_b - sum_x_b**2 / n_b) / (n_b - 1), 0.0)

    # Cohen's d (pooled) and Welch's t. NaN/Inf from zero-variance cases
    # is expected and handled by the caller's sort step; suppress warnings.
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        cohens_d = (mean_a - mean_b) / np.sqrt(pooled_var)
        se = np.sqrt(var_a / n_a + var_b / n_b)
        t = (mean_a - mean_b) / se

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "var_a": var_a,
        "var_b": var_b,
        "cohens_d": cohens_d,
        "t_statistic": t,
    }
