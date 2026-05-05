"""search_genes, gene_expression_summary, top_expressed_genes tools."""

from typing import Any

import numpy as np

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.parallel import reduce_gene_columns
from cell_explorer_agent.tools.registry import Tool
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
    z: ZarrAccess, *, limit_bytes: int, concurrency: int,
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
        if not mask.any():
            return {"error": f"group {group_value!r} is empty"}

        names = await z.var_names()

        def _reduce(gene: str, expr: np.ndarray) -> tuple[str, float]:
            return (gene, float(expr[mask].mean()))

        try:
            means = await reduce_gene_columns(
                z, names, _reduce, max_concurrency=concurrency,
            )
        except KeyError as exc:
            return {"error": f"gene {exc!s} not found"}
        except Exception as exc:
            return {"error": f"failed to read expression data: {exc}"}

        means.sort(key=lambda x: -x[1])
        top = means[:n]

        return cap_json_bytes(
            {
                "group_value": group_value,
                "genes": [{"symbol": g, "mean": m} for g, m in top],
            },
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="top_expressed_genes",
        kind="data",
        description=(
            "Return the top-N genes by mean expression within a group. "
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
