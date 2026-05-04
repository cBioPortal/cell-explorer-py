"""describe_obs_column and cluster_stats tools."""

from collections import Counter
from typing import Any

import numpy as np

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess

TOP_N = 50


def describe_obs_column_tool(z: ZarrAccess, *, limit_bytes: int) -> Tool:
    async def run(name: str) -> dict[str, Any]:
        try:
            col = await z.obs_column(name)
        except KeyError:
            return {"error": f"obs column {name!r} not found"}

        total = int(len(col.values))
        if col.dtype == "categorical":
            assert col.categories is not None
            counts = Counter(col.values.tolist())
            items = counts.most_common(TOP_N)
            other = total - sum(c for _, c in items)
            return cap_json_bytes(
                {
                    "dtype": "categorical",
                    "total": total,
                    "top_categories": [
                        {"value": col.categories[code], "count": int(count)}
                        for code, count in items
                    ],
                    "other_count": int(other),
                },
                limit_bytes=limit_bytes,
            )

        vals = np.asarray(col.values, dtype="float64")
        return cap_json_bytes(
            {
                "dtype": "numeric",
                "total": total,
                "stats": {
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "q1": float(np.quantile(vals, 0.25)),
                    "q3": float(np.quantile(vals, 0.75)),
                    "stddev": float(np.std(vals)),
                },
            },
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="describe_obs_column",
        kind="data",
        description=(
            "Describe one obs column. For categorical columns, returns the top 50 "
            "categories with counts plus the remainder. For numeric columns, returns "
            "min/max/mean/median/quartiles/stddev."
        ),
        args_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        func=run,
    )


def cluster_stats_tool(z: ZarrAccess, *, limit_bytes: int) -> Tool:
    async def run(obs_column: str) -> dict[str, Any]:
        try:
            col = await z.obs_column(obs_column)
        except KeyError:
            return {"error": f"obs column {obs_column!r} not found"}
        if col.dtype != "categorical" or col.categories is None:
            return {"error": f"{obs_column!r} is not categorical"}

        total = int(len(col.values))
        counts = Counter(col.values.tolist())
        items = counts.most_common(TOP_N)
        return cap_json_bytes(
            {
                "total": total,
                "groups": [
                    {
                        "value": col.categories[code],
                        "count": int(c),
                        "fraction": round(c / total, 4),
                    }
                    for code, c in items
                ],
            },
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="cluster_stats",
        kind="data",
        description=(
            "Count cells per group for a categorical obs column. Returns top 50 groups."
        ),
        args_schema={
            "type": "object",
            "properties": {"obs_column": {"type": "string"}},
            "required": ["obs_column"],
            "additionalProperties": False,
        },
        func=run,
    )
