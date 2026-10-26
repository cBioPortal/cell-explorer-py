"""plot_bar — a grouped bar chart comparing a few groups of an obs column.

The agent picks what to measure; the numbers are computed here from the
data, never supplied by the model. Output shape (shared by all measures):

    categories: x-axis items (cell types, genes, or the groups themselves)
    series:     one bar per series within each category (the compared groups)
    values:     values[i][j] for categories[i], series[j]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.data.genes import _xscan_panel_sums
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ObsColumn, ZarrAccess

MEASURES = ("composition", "cell_counts", "gene_mean")
GROUPS_CAP = 8
GENES_CAP = 10
COMPOSITION_TOP = 10
OTHER_LABEL = "All other"


class _ArgError(Exception):
    pass


async def _categorical(z: ZarrAccess, name: str, role: str) -> ObsColumn:
    try:
        col = await z.obs_column(name)
    except KeyError:
        raise _ArgError(f"{role} column {name!r} not found") from None
    if col.dtype != "categorical" or col.categories is None:
        raise _ArgError(f"{role} column {name!r} is not categorical")
    return col


def _code(col: ObsColumn, value: str, role: str) -> int:
    assert col.categories is not None
    if value not in col.categories:
        shown = ", ".join(map(repr, col.categories[:20]))
        raise _ArgError(
            f"{role} value {value!r} not in {col.name!r}; values include: {shown}"
        )
    return col.categories.index(value)


async def _where_mask(z: ZarrAccess, where: dict[str, str], n_obs: int) -> np.ndarray:
    mask = np.ones(n_obs, dtype=bool)
    for name, value in where.items():
        col = await _categorical(z, name, "where")
        mask &= col.values == _code(col, value, "where")
    return mask


def plot_bar_tool(z: ZarrAccess, *, limit_bytes: int, concurrency: int) -> Tool:
    async def run(
        measure: str,
        obs_column: str,
        groups: list[str],
        of: str | None = None,
        genes: list[str] | None = None,
        where: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            data = await _compute(
                measure, obs_column, groups, of, genes, where or {},
            )
        except _ArgError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"failed to read data: {exc}"}
        return cap_json_bytes(
            {"data": data, "chart": {"type": "grouped_bar", "data": data}},
            limit_bytes=limit_bytes,
        )

    async def _compute(
        measure: str,
        obs_column: str,
        groups: list[str],
        of: str | None,
        genes: list[str] | None,
        where: dict[str, str],
    ) -> dict[str, Any]:
        if measure not in MEASURES:
            raise _ArgError(f"measure must be one of {list(MEASURES)}")
        groups = list(dict.fromkeys(groups or []))
        if not groups or len(groups) > GROUPS_CAP:
            raise _ArgError(f"groups must list 1-{GROUPS_CAP} values of {obs_column!r}")

        col = await _categorical(z, obs_column, "obs")
        base = await _where_mask(z, where, len(col.values))
        masks = [base & (col.values == _code(col, g, "group")) for g in groups]
        n_cells = [int(m.sum()) for m in masks]

        out: dict[str, Any] = {
            "measure": measure,
            "obs_column": obs_column,
            "where": where or None,
            "n_cells": n_cells,
        }

        if measure == "cell_counts":
            return {
                **out,
                "title": f"Cells per {obs_column}",
                "value_kind": "count",
                "categories": groups,
                "series": ["cells"],
                "values": [[n] for n in n_cells],
            }

        if measure == "composition":
            if not of:
                raise _ArgError("composition needs `of`: the obs column to break groups down by")
            if of == obs_column:
                raise _ArgError("`of` must be a different column from obs_column")
            of_col = await _categorical(z, of, "of")
            assert of_col.categories is not None
            n_of = len(of_col.categories)
            counts = np.stack([
                np.bincount(of_col.values[m & (of_col.values >= 0)], minlength=n_of)
                for m in masks
            ])  # (groups, of-categories)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = counts / np.array(n_cells)[:, None]
            frac = np.nan_to_num(frac).T  # (of-categories, groups)

            keep = [i for i in range(n_of) if counts[:, i].any()]
            keep.sort(key=lambda i: -frac[i].sum())
            categories = [of_col.categories[i] for i in keep]
            values = [frac[i].tolist() for i in keep]
            if len(keep) > COMPOSITION_TOP:
                tail = keep[COMPOSITION_TOP:]
                categories = categories[:COMPOSITION_TOP] + [OTHER_LABEL]
                values = values[:COMPOSITION_TOP] + [frac[tail].sum(axis=0).tolist()]
            return {
                **out,
                "of": of,
                "title": f"{of} composition by {obs_column}",
                "value_kind": "fraction",
                "categories": categories,
                "series": groups,
                "values": values,
            }

        # gene_mean
        genes = list(dict.fromkeys(genes or []))
        if not genes or len(genes) > GENES_CAP:
            raise _ArgError(f"gene_mean needs `genes`: 1-{GENES_CAP} gene symbols")
        var_set = set(await z.var_names())
        missing = [g for g in genes if g not in var_set]
        if missing:
            raise _ArgError(f"genes not found in dataset: {missing[:5]}")
        sum_x, nnz, _ = await _xscan_panel_sums(z, genes, masks, concurrency=concurrency)
        denom = np.maximum(np.array(n_cells), 1)
        return {
            **out,
            "title": f"Mean expression by {obs_column}",
            "value_kind": "mean_expression",
            "categories": genes,
            "series": groups,
            "values": (sum_x / denom).tolist(),
            "fraction_expressing": (nnz / denom).tolist(),
        }

    return Tool(
        name="plot_bar",
        kind="data",
        description=(
            "Draw a grouped bar chart comparing a few groups (1-8 values of a "
            "categorical obs column). Use when the user asks to compare groups "
            "visually or asks for a bar chart. Measures: "
            "'composition' = fraction of each `of` category within each group "
            "(e.g. cell-type mix in Adnexa vs Omentum); "
            "'cell_counts' = number of cells per group; "
            "'gene_mean' = mean expression of 1-10 `genes` per group. "
            "`where` restricts all counts to cells matching every "
            "{obs_column: value} pair (e.g. only CD45+ sorted cells), and "
            "fractions are computed within that subset."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "measure": {"type": "string", "enum": list(MEASURES)},
                "obs_column": {"type": "string"},
                "groups": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": GROUPS_CAP,
                },
                "of": {"type": ["string", "null"]},
                "genes": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "maxItems": GENES_CAP,
                },
                "where": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["measure", "obs_column", "groups"],
            "additionalProperties": False,
        },
        func=run,
    )
