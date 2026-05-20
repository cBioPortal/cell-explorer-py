"""search_genes, gene_expression_summary, top_expressed_genes tools."""

import asyncio
from typing import Any

import numpy as np

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.parallel import reduce_gene_columns
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.strata_protocol import StrataAccess
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess

HARD_LIMIT = 50
FRACTION_THRESHOLD = 0.0  # considered "expressing" if > this


def search_genes_tool(z: ZarrAccess, *, limit_bytes: int) -> Tool:
    async def run(query: str, limit: int = 20) -> dict[str, Any]:
        capped = limit > HARD_LIMIT
        n = min(limit, HARD_LIMIT)
        names = await z.var_names()
        q = query.lower()
        matches = [
            {"symbol": name, "index": i}
            for i, name in enumerate(names)
            if q in name.lower()
        ][:n]
        return cap_json_bytes(
            {"matches": matches, "capped": capped},
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="search_genes",
        kind="data",
        description=(
            "Find gene symbols matching a substring (case-insensitive). "
            "Returns up to 50 matches."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        func=run,
    )


def _summary_stats(expr: np.ndarray) -> dict[str, float]:
    exp_mask = expr > FRACTION_THRESHOLD
    frac = float(exp_mask.mean()) if expr.size else 0.0
    return {
        "mean": float(expr.mean()) if expr.size else 0.0,
        "fraction_expressing": frac,
    }


def gene_expression_summary_tool(z: ZarrAccess, *, limit_bytes: int) -> Tool:
    async def run(gene: str, group_by: str | None = None) -> dict[str, Any]:
        try:
            expr = await z.gene_column(gene)
        except KeyError:
            return {"error": f"gene {gene!r} not found"}

        result: dict[str, Any] = {
            "gene": gene,
            "overall": _summary_stats(expr),
        }

        if group_by is not None:
            try:
                col = await z.obs_column(group_by)
            except KeyError:
                return {"error": f"obs column {group_by!r} not found"}
            if col.dtype != "categorical" or col.categories is None:
                return {"error": f"{group_by!r} is not categorical"}

            per_group: list[dict[str, Any]] = []
            codes = col.values
            for code, name in enumerate(col.categories):
                mask = codes == code
                if not mask.any():
                    continue
                stats = _summary_stats(expr[mask])
                per_group.append({"value": name, "count": int(mask.sum()), **stats})

            per_group.sort(key=lambda g: -g["mean"])
            result["per_group"] = per_group[:HARD_LIMIT]

        return cap_json_bytes(result, limit_bytes=limit_bytes)

    return Tool(
        name="gene_expression_summary",
        kind="data",
        description=(
            "Return overall mean + fraction expressing for one gene; "
            "if group_by is a categorical obs column, also return per-group stats "
            "for the top 50 groups by mean."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "gene": {"type": "string"},
                "group_by": {"type": ["string", "null"]},
            },
            "required": ["gene"],
            "additionalProperties": False,
        },
        func=run,
    )


def top_expressed_genes_tool(
    z: ZarrAccess,
    *,
    limit_bytes: int,
    concurrency: int,
    strata: StrataAccess | None = None,
) -> Tool:
    async def run(obs_column: str, group_value: str, n: int = 20) -> dict[str, Any]:
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
        mask = col.values == code
        n_cells = int(mask.sum())
        if n_cells == 0:
            return {"error": f"group {group_value!r} is empty"}

        names = await z.var_names()

        # Strata-first routing, same precedence compare_groups uses:
        # coarse -> atomic -> X-scan. Strata paths return per-gene sums for
        # all cells in the group; X-scan computes them inline.
        coarse_result = await _maybe_coarse_group_sums(
            strata, obs_column, group_value,
        )
        if coarse_result is not None:
            method = "via_coarse_strata"
            sum_x, nnz = coarse_result
        else:
            atomic_result = await _maybe_atomic_group_sums(
                strata, obs_column, group_value,
            )
            if atomic_result is not None:
                method = "via_atomic_strata"
                sum_x, nnz = atomic_result
            else:
                method = "via_xscan"
                try:
                    sum_x, nnz = await _xscan_group_sums(
                        z, names, mask, concurrency=concurrency,
                    )
                except KeyError as exc:
                    return {"error": f"gene {exc!s} not found"}
                except Exception as exc:
                    return {"error": f"failed to read expression data: {exc}"}

        means = sum_x.astype("float64", copy=False) / n_cells
        fractions = nnz.astype("float64", copy=False) / n_cells

        # np.argsort sorts NaN to the end; pick top-n by descending mean.
        keys = np.where(np.isfinite(means), -means, np.inf)
        order = np.argsort(keys, kind="stable")[:n]

        genes_list = [
            {
                "symbol": str(names[int(i)]),
                "mean": float(means[i]),
                "fraction_expressing": float(fractions[i]),
            }
            for i in order
        ]
        payload = {
            "obs_column": obs_column,
            "group_value": group_value,
            "method": method,
            "n_cells": n_cells,
            "genes": genes_list,
            # Inline-chart hint. Frontend reads chart.type to pick a renderer
            # and chart.data for the values to render. LLM-side, chart.data is
            # stripped in agent.py before being added to the tool_result block.
            "chart": {
                "type": "top_genes_bar",
                "data": {
                    "obs_column": obs_column,
                    "group_value": group_value,
                    "genes": genes_list,
                },
            },
        }
        return cap_json_bytes(payload, limit_bytes=limit_bytes)

    return Tool(
        name="top_expressed_genes",
        kind="data",
        description=(
            "Return the top-N genes by mean expression within a group. "
            "Output per gene: symbol, mean, fraction_expressing (nnz / n_cells). "
            "n <= 50."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "obs_column": {"type": "string"},
                "group_value": {"type": "string"},
                "n": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["obs_column", "group_value"],
            "additionalProperties": False,
        },
        func=run,
    )


async def _maybe_coarse_group_sums(
    strata: StrataAccess | None,
    obs_column: str,
    group_value: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (sum_x, nnz) for cells matching obs_column == group_value from
    the first coarse strata table whose axes == [obs_column], or None if no
    matching coarse exists or the group isn't present.
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
    return coarse.sum_x[idx], coarse.nnz[idx]


async def _maybe_atomic_group_sums(
    strata: StrataAccess | None,
    obs_column: str,
    group_value: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Aggregate atomic strata rows by an axis value to derive per-group sums.

    Two-phase fetch: read stratum_keys to find the rows matching group_value,
    then request only those rows of sum_x / nnz. Sums are additive so the
    result equals the coarse table on that axis (if it existed).
    """
    if strata is None or not strata.has_atomic():
        return None
    axes = strata.atomic_axes() or []
    if obs_column not in axes:
        return None
    axis_idx = axes.index(obs_column)

    keys, _n_cells = await strata.read_atomic_stratum_keys()
    rows = np.where(keys[:, axis_idx] == group_value)[0].tolist()
    if not rows:
        return None

    slim = await strata.read_atomic_rows(rows)
    return slim.sum_x.sum(axis=0), slim.nnz.sum(axis=0)


async def _xscan_group_sums(
    z: ZarrAccess,
    names: list[str],
    mask: np.ndarray,
    *,
    concurrency: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch each gene column with bounded concurrency, accumulate sum_x and
    nnz for the masked cells. Mirrors compare._xscan_sums but for a single
    group.
    """
    n_genes = len(names)
    sum_x = np.zeros(n_genes, dtype="float64")
    nnz = np.zeros(n_genes, dtype="int64")
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int, gene: str) -> None:
        async with sem:
            expr = await z.gene_column(gene)
        sub = expr[mask].astype("float64", copy=False)
        sum_x[i] = sub.sum()
        nnz[i] = int(np.count_nonzero(sub))

    await asyncio.gather(*(_one(i, g) for i, g in enumerate(names)))
    return sum_x, nnz


async def _xscan_panel_sums(
    z: ZarrAccess,
    names: list[str],
    masks: list[np.ndarray],
    *,
    concurrency: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-(gene, category) sum_x + nnz + per-category n_cells.

    Mirrors _xscan_group_sums but with K masks (one per category) instead of
    one. Single zarr fetch per gene; we slice into each category's mask after
    the fetch lands.

    Returns:
        sum_x: (n_genes, n_categories) float64
        nnz:   (n_genes, n_categories) int64
        n_cells: (n_categories,) int64
    """
    n_genes = len(names)
    n_cat = len(masks)
    sum_x = np.zeros((n_genes, n_cat), dtype="float64")
    nnz = np.zeros((n_genes, n_cat), dtype="int64")
    n_cells = np.array([int(m.sum()) for m in masks], dtype="int64")
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int, gene: str) -> None:
        async with sem:
            expr = await z.gene_column(gene)
        for j, mask in enumerate(masks):
            sub = expr[mask].astype("float64", copy=False)
            sum_x[i, j] = sub.sum()
            nnz[i, j] = int(np.count_nonzero(sub))

    await asyncio.gather(*(_one(i, g) for i, g in enumerate(names)))
    return sum_x, nnz, n_cells


PANEL_GENE_CAP = 50


def gene_panel_by_obs_tool(
    z: ZarrAccess,
    *,
    limit_bytes: int,
    concurrency: int,
) -> Tool:
    """Per-(gene, category) mean expression and fraction-expressing matrix.

    Returns data shaped for the frontend's ExpressionDotPlot renderer:
    rows = genes (in argument order), cols = categories (in dataset order).
    """

    async def run(obs_column: str, genes: list[str]) -> dict[str, Any]:
        if not isinstance(genes, list) or len(genes) == 0:
            return {"error": "genes must be a non-empty list of gene symbols"}
        if len(genes) > PANEL_GENE_CAP:
            return {
                "error": (
                    f"too many genes ({len(genes)} > {PANEL_GENE_CAP}); "
                    "narrow the panel and retry"
                )
            }

        try:
            col = await z.obs_column(obs_column)
        except KeyError:
            return {"error": f"obs column {obs_column!r} not found"}
        if col.dtype != "categorical" or col.categories is None:
            return {"error": f"{obs_column!r} is not categorical"}

        categories = list(col.categories)
        masks = [col.values == code for code in range(len(categories))]

        try:
            var_names = await z.var_names()
        except Exception as exc:
            return {"error": f"failed to read var names: {exc}"}
        var_set = set(var_names)
        missing = [g for g in genes if g not in var_set]
        if missing:
            return {"error": f"genes not found in dataset: {missing[:5]}"}

        try:
            sum_x, nnz, n_cells = await _xscan_panel_sums(
                z, genes, masks, concurrency=concurrency,
            )
        except KeyError as exc:
            return {"error": f"gene {exc!s} not found"}
        except Exception as exc:
            return {"error": f"failed to read expression data: {exc}"}

        # Avoid division by zero for empty categories.
        with np.errstate(divide="ignore", invalid="ignore"):
            n_cells_safe = np.where(n_cells == 0, 1, n_cells)
            mean = sum_x.astype("float64") / n_cells_safe
            frac = nnz.astype("float64") / n_cells_safe
            mean[:, n_cells == 0] = 0.0
            frac[:, n_cells == 0] = 0.0

        data = {
            "genes": list(genes),
            "categories": categories,
            "obs_column": obs_column,
            "mean": mean.tolist(),
            "frac_expressing": frac.tolist(),
            "n_cells_per_category": n_cells.tolist(),
        }
        return cap_json_bytes(
            {
                "data": data,
                "chart": {"type": "gene_panel_dotplot", "data": data},
            },
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="gene_panel_by_obs",
        kind="data",
        description=(
            "Per-(gene, category) mean expression and fraction-expressing for "
            "a panel of genes across a categorical obs column. Use when the "
            "user wants to see marker genes / cell-type-resolved expression as "
            "a dotplot (e.g. 'compare CD8A, CD4, GZMB across cell types'). "
            "Genes must already be resolved to canonical symbols/IDs (use "
            "search_genes first if needed). Max 50 genes."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "obs_column": {"type": "string"},
                "genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": PANEL_GENE_CAP,
                },
            },
            "required": ["obs_column", "genes"],
            "additionalProperties": False,
        },
        func=run,
    )
