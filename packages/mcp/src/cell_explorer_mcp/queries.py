"""Dataset queries behind the MCP tools.

Data-volume policy: these run on a shared server against stores of up to
millions of cells, so every query is bounded.

- Grouped statistics come from precomputed coarse strata tables (per-group
  sum_x / nnz / n_cells for every gene) whenever one exists for the column.
- Reading one gene column of X is allowed up to MAX_X_COLUMN_CELLS cells, and
  at most X_READ_CONCURRENCY such reads run at once across the server.
- Scanning all of X (top genes without strata) is allowed only up to
  MAX_X_SCAN_ELEMENTS cells x genes; larger requests are refused with a hint
  naming the columns that strata can answer.

There is deliberately no differential-expression tool: an X-based DE reads the
whole matrix. A strata-based one (Welch t-test from sum_x / sum_xx / n_cells)
fits within this policy if it is added.
"""

import asyncio
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from cell_explorer_mcp.zarr_cache import OpenDataset

MAX_RESULTS = 50
MAX_X_COLUMN_CELLS = 2_000_000
MAX_X_SCAN_ELEMENTS = 250_000_000
X_READ_CONCURRENCY = 2
# Gene columns per X read during a full scan; matches the published stores'
# 30-gene chunk width.
X_SCAN_BLOCK = 30

_x_semaphore = asyncio.Semaphore(X_READ_CONCURRENCY)


class QueryError(ValueError):
    """A request the dataset can't answer; the message is shown to the caller."""


def _round(x: float) -> float:
    return float(round(float(x), 4))


async def _resolve_gene(ds: OpenDataset, gene: str) -> int:
    try:
        return await ds.gene_index(gene)
    except KeyError:
        suggestions: list[dict[str, str]] = []
        # Shorten the query until something matches, so typos still get hints.
        for end in range(len(gene), 1, -1):
            suggestions = (await search_genes(ds, gene[:end], limit=5))["matches"]
            if suggestions:
                break
        hint = f" Similar: {', '.join(m['symbol'] for m in suggestions)}." if suggestions else ""
        raise QueryError(f"Gene {gene!r} not found in dataset {ds.slug!r}.{hint}") from None


async def _categorical(ds: OpenDataset, name: str) -> pd.Categorical:
    try:
        col = await ds.obs_column(name)
    except KeyError:
        raise QueryError(
            f"Obs column {name!r} not found. Available: {', '.join(ds.anndata.obs_columns)}"
        ) from None
    if not isinstance(col, pd.Categorical):
        raise QueryError(f"Obs column {name!r} is not categorical.")
    return col


def _strata_hint(ds: OpenDataset) -> str:
    columns = sorted({ds.strata.coarse_axes(s)[0] for s in ds.strata.coarse_slugs()
                      if len(ds.strata.coarse_axes(s)) == 1})
    return f" Columns with precomputed group stats: {', '.join(columns)}." if columns else ""


async def _read_gene_column(ds: OpenDataset, gene_idx: int) -> np.ndarray:
    if ds.n_obs > MAX_X_COLUMN_CELLS:
        raise QueryError(
            f"Dataset {ds.slug!r} has {ds.n_obs:,} cells; per-cell reads are limited to "
            f"{MAX_X_COLUMN_CELLS:,}.{_strata_hint(ds)}"
        )
    async with _x_semaphore:
        return await ds.gene_column(gene_idx)


# --- dataset info ------------------------------------------------------------


async def dataset_info(ds: OpenDataset, entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata") or {}
    obs_meta = {c["name"]: c for c in metadata.get("obs_columns") or []}
    obs_columns = []
    for name in ds.anndata.obs_columns:
        meta = obs_meta.get(name, {})
        col: dict[str, Any] = {"name": name, "dtype": meta.get("dtype")}
        if meta.get("cardinality") is not None:
            col["n_categories"] = meta["cardinality"]
        if meta.get("values"):
            col["categories"] = meta["values"]
        obs_columns.append(col)

    coarse = [ds.strata.coarse_axes(s) for s in ds.strata.coarse_slugs()]
    return {
        "slug": ds.slug,
        "name": entry.get("name"),
        "description": entry.get("description"),
        "collection": entry.get("collection"),
        "n_cells": ds.n_obs,
        "n_genes": ds.n_vars,
        "gene_symbol_column": await ds.symbol_column(),
        "obs_columns": obs_columns,
        "var_columns": ds.anndata.var_columns,
        "embeddings": ds.anndata.obsm_keys,
        "fast_group_by_columns": [axes[0] for axes in coarse if len(axes) == 1],
    }


# --- genes -------------------------------------------------------------------


async def search_genes(ds: OpenDataset, query: str, limit: int = 20) -> dict[str, Any]:
    """Case-insensitive match on symbol or var ID; exact, then prefix, then substring."""
    limit = max(1, min(limit, MAX_RESULTS))
    q = query.strip().lower()
    if not q:
        raise QueryError("query must not be empty")
    symbols = await ds.gene_symbols()
    ids = await ds.var_ids()
    ranked: list[tuple[int, int, int]] = []
    for i, (sym, gid) in enumerate(zip(symbols, ids)):
        s, g = sym.lower(), gid.lower()
        if q == s or q == g:
            rank = 0
        elif s.startswith(q) or g.startswith(q):
            rank = 1
        elif q in s or q in g:
            rank = 2
        else:
            continue
        ranked.append((rank, len(sym), i))
    ranked.sort()
    matches = [
        {"symbol": symbols[i], **({"id": ids[i]} if ids[i] != symbols[i] else {})}
        for _, _, i in ranked[:limit]
    ]
    return {"query": query, "n_matches": len(ranked), "matches": matches}


def _stats(sum_x: float, nnz: int, n_cells: int) -> dict[str, Any]:
    if n_cells == 0:
        return {"n_cells": 0, "mean": 0.0, "fraction_expressing": 0.0}
    return {
        "n_cells": int(n_cells),
        "mean": _round(sum_x / n_cells),
        "fraction_expressing": _round(nnz / n_cells),
    }


async def gene_expression(
    ds: OpenDataset, gene: str, group_by: str | None = None
) -> dict[str, Any]:
    gene_idx = await _resolve_gene(ds, gene)
    symbol = (await ds.gene_symbols())[gene_idx]
    result: dict[str, Any] = {"gene": symbol}

    slug = ds.coarse_slug_for(group_by) if group_by else None
    if group_by is None and ds.strata.coarse_slugs():
        # Any single coarse table sums over every cell, so it gives the
        # overall stats without touching X.
        slug = min(ds.strata.coarse_slugs(), key=ds.strata.coarse_strata_count)

    if slug is not None:
        table = await ds.coarse_table(slug)
        sum_x = table.sum_x[:, gene_idx].astype("float64")
        nnz = table.nnz[:, gene_idx].astype("int64")
        n_cells = table.n_cells.astype("int64")
        result["source"] = "strata"
        result["overall"] = _stats(sum_x.sum(), nnz.sum(), n_cells.sum())
        if group_by is not None:
            groups = [
                {"group": str(table.stratum_keys[i, 0]), **_stats(sum_x[i], nnz[i], n_cells[i])}
                for i in range(len(n_cells))
                if n_cells[i] > 0
            ]
            result["group_by"] = group_by
            result["groups"] = sorted(groups, key=lambda g: -g["mean"])[:MAX_RESULTS]
        return result

    categories = await _categorical(ds, group_by) if group_by else None
    expr = (await _read_gene_column(ds, gene_idx)).astype("float64", copy=False)
    expressing = expr > 0
    result["source"] = "cells"
    result["overall"] = _stats(expr.sum(), int(expressing.sum()), expr.size)
    if categories is not None:
        codes = np.asarray(categories.codes)
        valid = codes >= 0
        n_cat = len(categories.categories)
        n_cells = np.bincount(codes[valid], minlength=n_cat)
        sum_x = np.bincount(codes[valid], weights=expr[valid], minlength=n_cat)
        nnz = np.bincount(codes[valid], weights=expressing[valid], minlength=n_cat)
        groups = [
            {"group": str(name), **_stats(sum_x[i], int(nnz[i]), int(n_cells[i]))}
            for i, name in enumerate(categories.categories)
            if n_cells[i] > 0
        ]
        result["group_by"] = group_by
        result["n_groups"] = len(groups)
        result["groups"] = sorted(groups, key=lambda g: -g["mean"])[:MAX_RESULTS]
    return result


async def top_genes(
    ds: OpenDataset, obs_column: str, group_value: str, n: int = 20
) -> dict[str, Any]:
    n = max(1, min(n, MAX_RESULTS))
    slug = ds.coarse_slug_for(obs_column)
    if slug is not None:
        table = await ds.coarse_table(slug)
        rows = np.where(table.stratum_keys[:, 0] == group_value)[0]
        if len(rows) == 0:
            values = ", ".join(str(v) for v in table.stratum_keys[:MAX_RESULTS, 0])
            raise QueryError(f"Group {group_value!r} not in {obs_column!r}. Values: {values}")
        row = int(rows[0])
        n_cells = int(table.n_cells[row])
        sum_x = table.sum_x[row].astype("float64")
        nnz = table.nnz[row].astype("int64")
        source = "strata"
    else:
        categories = await _categorical(ds, obs_column)
        if group_value not in categories.categories:
            values = ", ".join(str(v) for v in list(categories.categories)[:MAX_RESULTS])
            raise QueryError(f"Group {group_value!r} not in {obs_column!r}. Values: {values}")
        mask = np.asarray(categories.codes) == categories.categories.get_loc(group_value)
        n_cells = int(mask.sum())
        sum_x, nnz = await _scan_x_group_sums(ds, mask, obs_column)
        source = "cells"

    if n_cells == 0:
        raise QueryError(f"Group {group_value!r} in {obs_column!r} has no cells.")
    means = sum_x / n_cells
    order = np.argsort(-means, kind="stable")[:n]
    symbols = await ds.gene_symbols()
    return {
        "obs_column": obs_column,
        "group": group_value,
        "n_cells": n_cells,
        "source": source,
        "genes": [
            {
                "symbol": symbols[int(i)],
                "mean": _round(means[i]),
                "fraction_expressing": _round(nnz[i] / n_cells),
            }
            for i in order
        ],
    }


async def _scan_x_group_sums(
    ds: OpenDataset, mask: np.ndarray, obs_column: str
) -> tuple[np.ndarray, np.ndarray]:
    elements = ds.n_obs * ds.n_vars
    x = await ds.dense_x()
    if x is None or elements > MAX_X_SCAN_ELEMENTS:
        raise QueryError(
            f"Top genes by {obs_column!r} needs a scan of the full expression matrix "
            f"({ds.n_obs:,} cells x {ds.n_vars:,} genes), which exceeds this server's limit."
            f"{_strata_hint(ds)}"
        )
    sum_x = np.zeros(ds.n_vars, dtype="float64")
    nnz = np.zeros(ds.n_vars, dtype="int64")
    async with _x_semaphore:
        for start in range(0, ds.n_vars, X_SCAN_BLOCK):
            stop = min(start + X_SCAN_BLOCK, ds.n_vars)
            block = np.asarray(await x.getitem((slice(None), slice(start, stop))))
            sub = block[mask].astype("float64", copy=False)
            sum_x[start:stop] = sub.sum(axis=0)
            nnz[start:stop] = (sub > 0).sum(axis=0)
    return sum_x, nnz


# --- obs -----------------------------------------------------------------------


async def obs_column_values(ds: OpenDataset, column: str, limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(limit, MAX_RESULTS))
    try:
        col = await ds.obs_column(column)
    except KeyError:
        raise QueryError(
            f"Obs column {column!r} not found. Available: {', '.join(ds.anndata.obs_columns)}"
        ) from None

    if isinstance(col, pd.Categorical):
        counts = pd.Series(col).value_counts(dropna=False)
        total = int(counts.sum())
        values = [
            {"value": None if pd.isna(v) else str(v), "count": int(c), "fraction": _round(c / total)}
            for v, c in counts.items()
        ]
        return {
            "column": column,
            "dtype": "categorical",
            "n_cells": total,
            "n_values": len(values),
            "values": values[:limit],
            "other_count": int(sum(v["count"] for v in values[limit:])),
        }

    arr = np.asarray(col)
    if np.issubdtype(arr.dtype, np.number) and arr.dtype != bool:
        vals = arr.astype("float64")
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return {"column": column, "dtype": "numeric", "n_cells": int(vals.size), "n_missing": int(vals.size)}
        q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
        return {
            "column": column,
            "dtype": "numeric",
            "n_cells": int(vals.size),
            "n_missing": int(vals.size - finite.size),
            "stats": {
                "min": _round(finite.min()),
                "q1": _round(q1),
                "median": _round(median),
                "mean": _round(finite.mean()),
                "q3": _round(q3),
                "max": _round(finite.max()),
                "std": _round(finite.std()),
            },
        }

    counts = Counter(None if v is None else str(v) for v in arr.tolist())
    total = int(arr.size)
    items = counts.most_common()
    return {
        "column": column,
        "dtype": "string",
        "n_cells": total,
        "n_values": len(items),
        "values": [
            {"value": v, "count": int(c), "fraction": _round(c / total)} for v, c in items[:limit]
        ],
        "other_count": int(sum(c for _, c in items[limit:])),
    }
