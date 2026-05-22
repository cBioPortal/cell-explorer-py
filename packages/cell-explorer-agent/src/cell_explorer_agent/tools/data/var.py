"""describe_var_column tool — mirror of describe_obs_column but for var columns.

Var columns are the per-gene metadata columns in the AnnData var dataframe.
Typical examples: `gene_symbol`, `feature_id`, `highly_variable`,
`mean_counts`, `dispersions_norm`. They have the same categorical/numeric
shape as obs columns, so the tool output mirrors describe_obs_column exactly.
"""

from collections import Counter
from typing import Any

import numpy as np

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess

TOP_N = 50


def describe_var_column_tool(z: ZarrAccess, *, limit_bytes: int) -> Tool:
    async def run(name: str) -> dict[str, Any]:
        try:
            col = await z.var_column(name)
        except KeyError:
            return {"error": f"var column {name!r} not found"}

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
        name="describe_var_column",
        kind="data",
        description=(
            "Describe one var (per-gene) column. Use to inspect gene-metadata "
            "columns like 'gene_symbol' or 'feature_id'. For categorical "
            "columns, returns the top 50 values with counts plus the remainder. "
            "For numeric columns, returns min/max/mean/median/quartiles/stddev. "
            "Call get_dataset_schema first to learn which var columns are "
            "available."
        ),
        args_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        func=run,
    )
