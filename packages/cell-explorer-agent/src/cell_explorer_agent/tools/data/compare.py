"""compare_groups — rank genes by differential expression between two groups.

v1 uses Cohen's d (pooled-variance standardized mean difference) as the
primary ranking metric, with Welch's t-statistic returned alongside. Both
are scale-invariant in their ordering — they work correctly on raw counts,
log-normalized, z-scored, and Pearson-residual data.

Two internal paths:
  - Strata fast path: when a coarse strata table covers the queried
    obs_column, group means and variances come from precomputed sums.
  - X-scan fallback: when no matching coarse table exists (or no strata
    are wired in at all), read each gene column and compute the sums
    inline. Both paths funnel through compare_stats.compute_stats for
    byte-identical math.

Task 3 wires the strata path; Task 2 only implements the X-scan path
and the new output schema.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.data.compare_stats import compute_stats
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.strata_protocol import StrataAccess
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess

HARD_LIMIT = 50


def compare_groups_tool(
    z: ZarrAccess,
    *,
    limit_bytes: int,
    concurrency: int,
    strata: StrataAccess | None = None,
) -> Tool:
    async def run(
        obs_column: str,
        group_a: str,
        group_b: str,
        n: int = 20,
    ) -> dict[str, Any]:
        n = min(max(n, 1), HARD_LIMIT)
        try:
            col = await z.obs_column(obs_column)
        except KeyError:
            return {"error": f"obs column {obs_column!r} not found"}
        if col.dtype != "categorical" or col.categories is None:
            return {"error": f"{obs_column!r} is not categorical"}
        for g in (group_a, group_b):
            if g not in col.categories:
                return {"error": f"group {g!r} not in {obs_column!r}"}

        codes = col.values
        a_mask = codes == col.categories.index(group_a)
        b_mask = codes == col.categories.index(group_b)
        n_a = int(a_mask.sum())
        n_b = int(b_mask.sum())
        if n_a < 2 or n_b < 2:
            return {"error": "at least one group has < 2 cells; variance is undefined"}

        names = await z.var_names()

        # Try strata fast path first.
        strata_inputs = await _maybe_strata_sums(
            strata, obs_column, group_a, group_b,
        )
        if strata_inputs is not None:
            method = "via_coarse_strata"
            sum_x_a, sum_xx_a, sum_x_b, sum_xx_b = strata_inputs
            # n_a/n_b from the obs mask above are the reported group sizes —
            # we don't read coarse.n_cells[idx], so both paths report the
            # same denominator even if the strata table and the obs mask
            # could in principle diverge.
        else:
            method = "via_xscan"
            try:
                sum_x_a, sum_xx_a, sum_x_b, sum_xx_b = await _xscan_sums(
                    z, names, a_mask, b_mask, concurrency=concurrency,
                )
            except KeyError as exc:
                return {"error": f"gene {exc!s} not found"}
            except Exception as exc:
                return {"error": f"failed to read expression data: {exc}"}

        return _assemble_result(
            obs_column=obs_column,
            group_a=group_a,
            group_b=group_b,
            method=method,
            names=names,
            sum_x_a=sum_x_a, sum_xx_a=sum_xx_a, n_a=n_a,
            sum_x_b=sum_x_b, sum_xx_b=sum_xx_b, n_b=n_b,
            n=n,
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="compare_groups",
        kind="data",
        description=(
            "Rank the top N most differentially expressed genes between two "
            "groups of a categorical obs column. Primary metric is Cohen's d "
            "(pooled-variance standardized mean difference); Welch's t-statistic "
            "is reported alongside. Works on any expression scale (raw counts, "
            "log-normalized, z-scored). Uses precomputed strata tables when "
            "available for speed. n <= 50."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "obs_column": {"type": "string"},
                "group_a": {"type": "string"},
                "group_b": {"type": "string"},
                "n": {"type": "integer", "minimum": 1, "maximum": HARD_LIMIT},
            },
            "required": ["obs_column", "group_a", "group_b"],
            "additionalProperties": False,
        },
        func=run,
    )


async def _xscan_sums(
    z: ZarrAccess,
    names: list[str],
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    *,
    concurrency: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch each gene column with bounded concurrency and accumulate
    per-group (sum_x, sum_xx) into 1D arrays aligned with `names`.
    """
    n_genes = len(names)
    sum_x_a = np.zeros(n_genes, dtype="float64")
    sum_xx_a = np.zeros(n_genes, dtype="float64")
    sum_x_b = np.zeros(n_genes, dtype="float64")
    sum_xx_b = np.zeros(n_genes, dtype="float64")
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int, gene: str) -> None:
        async with sem:
            expr = await z.gene_column(gene)
        expr64 = expr.astype("float64", copy=False)
        a = expr64[a_mask]
        b = expr64[b_mask]
        sum_x_a[i] = a.sum()
        sum_xx_a[i] = (a * a).sum()
        sum_x_b[i] = b.sum()
        sum_xx_b[i] = (b * b).sum()

    await asyncio.gather(*(_one(i, g) for i, g in enumerate(names)))
    return sum_x_a, sum_xx_a, sum_x_b, sum_xx_b


def _assemble_result(
    *,
    obs_column: str,
    group_a: str,
    group_b: str,
    method: str,
    names: list[str],
    sum_x_a: np.ndarray,
    sum_xx_a: np.ndarray,
    n_a: int,
    sum_x_b: np.ndarray,
    sum_xx_b: np.ndarray,
    n_b: int,
    n: int,
    limit_bytes: int,
) -> dict[str, Any]:
    """Run the shared compute, sort by |cohens_d| with non-finite to end,
    take top-n, and wrap in the v1 output schema.
    """
    stats = compute_stats(
        sum_x_a, sum_xx_a, n_a, sum_x_b, sum_xx_b, n_b,
    )
    d = stats["cohens_d"]
    # Non-finite (NaN/Inf) sort keys go to +inf so they fall to the end.
    keys = np.where(np.isfinite(d), -np.abs(d), np.inf)
    order = np.argsort(keys, kind="stable")[:n]
    genes = [
        {
            "symbol": str(names[int(i)]),
            "cohens_d": float(d[i]),
            "t_statistic": float(stats["t_statistic"][i]),
            "mean_a": float(stats["mean_a"][i]),
            "mean_b": float(stats["mean_b"][i]),
            "var_a": float(stats["var_a"][i]),
            "var_b": float(stats["var_b"][i]),
            "n_a": n_a,
            "n_b": n_b,
        }
        for i in order
    ]
    return cap_json_bytes(
        {
            "obs_column": obs_column,
            "group_a": group_a,
            "group_b": group_b,
            "ranked_by": "cohens_d",
            "method": method,
            "genes": genes,
        },
        limit_bytes=limit_bytes,
    )


async def _maybe_strata_sums(
    strata: StrataAccess | None,
    obs_column: str,
    group_a: str,
    group_b: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return per-group (sum_x_a, sum_xx_a, sum_x_b, sum_xx_b) from the
    first coarse strata table whose axes == [obs_column], or None if no
    matching coarse table exists or the groups aren't present in it.
    """
    if strata is None:
        return None
    matching_slug: str | None = None
    for slug in strata.coarse_slugs():
        if strata.coarse_axes(slug) == [obs_column]:
            matching_slug = slug
            break
    if matching_slug is None:
        return None

    coarse = await strata.read_coarse(matching_slug)
    keys_col0 = coarse.stratum_keys[:, 0]
    a_rows = np.where(keys_col0 == group_a)[0]
    b_rows = np.where(keys_col0 == group_b)[0]
    if len(a_rows) == 0 or len(b_rows) == 0:
        # Group not present in the coarse table — fall back to X-scan.
        return None
    a_idx, b_idx = int(a_rows[0]), int(b_rows[0])
    return (
        coarse.sum_x[a_idx],
        coarse.sum_xx[a_idx],
        coarse.sum_x[b_idx],
        coarse.sum_xx[b_idx],
    )
