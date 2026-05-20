"""find_markers — rank genes that distinguish one group from the rest.

Implements the classic marker-gene query: given (obs_column, group_value),
compare the cells matching that filter against every other cell in the
dataset, rank by Cohen's d (primary) + Welch's t-statistic (secondary).

This is `compare_groups` with the second group computed implicitly as
"everything else": rest_sums = total_sums - group_sums. Sums are additive
so the result is identical to scanning every non-matching cell, but the
rest side comes from a one-time sum over the same coarse/atomic table the
group side is read from — no extra I/O.

Routing parallels compare_groups: coarse strata → atomic strata → X-scan.
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


def find_markers_tool(
    z: ZarrAccess,
    *,
    limit_bytes: int,
    concurrency: int,
    strata: StrataAccess | None = None,
) -> Tool:
    async def run(
        obs_column: str,
        group_value: str,
        n: int = 20,
    ) -> dict[str, Any]:
        n = min(max(n, 1), HARD_LIMIT)
        try:
            col = await z.obs_column(obs_column)
        except KeyError:
            return {"error": f"obs column {obs_column!r} not found"}
        if col.dtype != "categorical" or col.categories is None:
            return {"error": f"{obs_column!r} is not categorical"}
        if group_value not in col.categories:
            return {"error": f"group {group_value!r} not in {obs_column!r}"}

        code = col.categories.index(group_value)
        group_mask = col.values == code
        n_group = int(group_mask.sum())
        n_rest = int(len(col.values) - n_group)
        if n_group < 2 or n_rest < 2:
            return {"error": "group or rest has < 2 cells; variance is undefined"}

        names = await z.var_names()

        # Strata-first routing — coarse → atomic → X-scan, same precedence
        # compare_groups uses.
        coarse_result = await _maybe_coarse_sums(strata, obs_column, group_value)
        if coarse_result is not None:
            method = "via_coarse_strata"
            sum_x_g, sum_xx_g, sum_x_total, sum_xx_total = coarse_result
        else:
            atomic_result = await _maybe_atomic_sums(strata, obs_column, group_value)
            if atomic_result is not None:
                method = "via_atomic_strata"
                sum_x_g, sum_xx_g, sum_x_total, sum_xx_total = atomic_result
            else:
                method = "via_xscan"
                try:
                    sum_x_g, sum_xx_g, sum_x_total, sum_xx_total = await _xscan_sums(
                        z, names, group_mask, concurrency=concurrency,
                    )
                except KeyError as exc:
                    return {"error": f"gene {exc!s} not found"}
                except Exception as exc:
                    return {"error": f"failed to read expression data: {exc}"}

        # rest = total - group, element-wise (sums are additive).
        sum_x_r = sum_x_total - sum_x_g
        sum_xx_r = sum_xx_total - sum_xx_g

        stats = compute_stats(
            sum_x_g, sum_xx_g, n_group,
            sum_x_r, sum_xx_r, n_rest,
        )
        d = stats["cohens_d"]
        keys = np.where(np.isfinite(d), -np.abs(d), np.inf)
        order = np.argsort(keys, kind="stable")[:n]

        genes = [
            {
                "symbol": str(names[int(i)]),
                "cohens_d": float(d[i]),
                "t_statistic": float(stats["t_statistic"][i]),
                "mean_group": float(stats["mean_a"][i]),
                "mean_rest": float(stats["mean_b"][i]),
                "var_group": float(stats["var_a"][i]),
                "var_rest": float(stats["var_b"][i]),
                "n_group": n_group,
                "n_rest": n_rest,
            }
            for i in order
        ]
        return cap_json_bytes(
            {
                "obs_column": obs_column,
                "group_value": group_value,
                "ranked_by": "cohens_d",
                "method": method,
                "n_cells_group": n_group,
                "n_cells_rest": n_rest,
                "genes": genes,
            },
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="find_markers",
        kind="data",
        description=(
            "Return the top N marker genes for a group, ranked by Cohen's d "
            "vs all other cells in the dataset. Primary metric Cohen's d; "
            "Welch's t-statistic reported alongside. Works on any expression "
            "scale (raw counts, log-normalized, z-scored). Uses precomputed "
            "strata tables when available for speed. n <= 50."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "obs_column": {"type": "string"},
                "group_value": {"type": "string"},
                "n": {"type": "integer", "minimum": 1, "maximum": HARD_LIMIT},
            },
            "required": ["obs_column", "group_value"],
            "additionalProperties": False,
        },
        func=run,
    )


async def _maybe_coarse_sums(
    strata: StrataAccess | None,
    obs_column: str,
    group_value: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (sum_x_group, sum_xx_group, sum_x_total, sum_xx_total) from a
    coarse table whose axes == [obs_column], or None if not strata-resolvable.

    Totals come from summing all rows of the same coarse table — every cell
    in the dataset is partitioned by this single-axis coarse, so row-sums
    equal dataset-totals.
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
    rows = np.where(coarse.stratum_keys[:, 0] == group_value)[0]
    if len(rows) == 0:
        return None
    idx = int(rows[0])
    sum_x_total = coarse.sum_x.sum(axis=0)
    sum_xx_total = coarse.sum_xx.sum(axis=0)
    return coarse.sum_x[idx], coarse.sum_xx[idx], sum_x_total, sum_xx_total


async def _maybe_atomic_sums(
    strata: StrataAccess | None,
    obs_column: str,
    group_value: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Aggregate atomic rows by an axis value to derive group sums; totals
    come from summing all atomic rows. Sums are additive.
    """
    if strata is None or not strata.has_atomic():
        return None
    axes = strata.atomic_axes() or []
    if obs_column not in axes:
        return None
    axis_idx = axes.index(obs_column)

    keys, _n_cells = await strata.read_atomic_stratum_keys()
    group_rows = np.where(keys[:, axis_idx] == group_value)[0].tolist()
    if not group_rows:
        return None

    # Read the whole atomic — we need every row anyway (totals require it).
    # The row-selective path adds no value when we want both group and total.
    atomic = await strata.read_atomic()
    group_idx = np.asarray(group_rows, dtype=np.int64)
    sum_x_g = atomic.sum_x[group_idx].sum(axis=0)
    sum_xx_g = atomic.sum_xx[group_idx].sum(axis=0)
    sum_x_total = atomic.sum_x.sum(axis=0)
    sum_xx_total = atomic.sum_xx.sum(axis=0)
    return sum_x_g, sum_xx_g, sum_x_total, sum_xx_total


async def _xscan_sums(
    z: ZarrAccess,
    names: list[str],
    group_mask: np.ndarray,
    *,
    concurrency: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch each gene column with bounded concurrency. Compute sum_x and
    sum_xx for the group AND totals in one pass (totals are free — we
    already have the full column in memory).
    """
    n_genes = len(names)
    sum_x_g = np.zeros(n_genes, dtype="float64")
    sum_xx_g = np.zeros(n_genes, dtype="float64")
    sum_x_total = np.zeros(n_genes, dtype="float64")
    sum_xx_total = np.zeros(n_genes, dtype="float64")
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int, gene: str) -> None:
        async with sem:
            expr = await z.gene_column(gene)
        expr64 = expr.astype("float64", copy=False)
        sum_x_total[i] = expr64.sum()
        sum_xx_total[i] = (expr64 * expr64).sum()
        g = expr64[group_mask]
        sum_x_g[i] = g.sum()
        sum_xx_g[i] = (g * g).sum()

    await asyncio.gather(*(_one(i, g) for i, g in enumerate(names)))
    return sum_x_g, sum_xx_g, sum_x_total, sum_xx_total
