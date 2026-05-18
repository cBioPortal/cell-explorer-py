"""compare_groups — fast log2-fold-change between two categorical groups.

This is a heuristic, not a statistical DE test. Expression data is stored float16
(already lossy), so a fast approximation is appropriate for v1.
"""

from typing import Any

import numpy as np

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.parallel import reduce_gene_columns
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess

HARD_LIMIT = 50
PSEUDO = 1e-3  # pseudo-count for log ratio


def compare_groups_tool(
    z: ZarrAccess, *, limit_bytes: int, concurrency: int,
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
        if not a_mask.any() or not b_mask.any():
            return {"error": "at least one group is empty"}

        names = await z.var_names()

        def _reduce(gene: str, expr: np.ndarray) -> tuple[str, float, float, float]:
            a_mean = float(expr[a_mask].mean())
            b_mean = float(expr[b_mask].mean())
            lfc = float(np.log2((a_mean + PSEUDO) / (b_mean + PSEUDO)))
            return (gene, lfc, a_mean, b_mean)

        try:
            results = await reduce_gene_columns(
                z, names, _reduce, max_concurrency=concurrency,
            )
        except KeyError as exc:
            return {"error": f"gene {exc!s} not found"}
        except Exception as exc:
            return {"error": f"failed to read expression data: {exc}"}

        # np.argsort sorts NaN to the end, so genes with undefined LFC
        # (negative ratio after pseudocount on scaled data) cannot displace
        # finite top-magnitude entries. Python's list.sort with NaN keys is
        # undefined and silently corrupted the ranking.
        lfc_arr = np.array([r[1] for r in results], dtype="float64")
        order = np.argsort(-np.abs(lfc_arr))
        top = [results[int(i)] for i in order[:n]]

        return cap_json_bytes(
            {
                "group_a": group_a,
                "group_b": group_b,
                "method": "log2_fold_change_fast_heuristic",
                "genes": [
                    {
                        "symbol": s,
                        "log2_fold_change": lfc,
                        "mean_a": a,
                        "mean_b": b,
                    }
                    for s, lfc, a, b in top
                ],
            },
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="compare_groups",
        kind="data",
        description=(
            "Rank genes by |log2 fold change| between two groups of a categorical "
            "obs column. Fast heuristic; not a statistical DE test. n ≤ 50."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "obs_column": {"type": "string"},
                "group_a": {"type": "string"},
                "group_b": {"type": "string"},
                "n": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["obs_column", "group_a", "group_b"],
            "additionalProperties": False,
        },
        func=run,
    )
