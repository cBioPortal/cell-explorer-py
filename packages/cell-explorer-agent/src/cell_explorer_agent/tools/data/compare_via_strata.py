"""compare_groups_via_strata — log2-fold-change between two groups, via coarse strata.

Mirrors compare_groups's API exactly. The strata-powered path: looks up the
coarse table whose axes match [obs_column], reads sums + cell counts, computes
LFC from per-stratum means without scanning X. Errors when no matching coarse
exists — caller falls back to compare_groups.
"""

from typing import Any

import numpy as np

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.strata_protocol import StrataAccess
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess

HARD_LIMIT = 50
PSEUDO = 1e-3  # pseudo-count for log ratio (matches compare_groups)


def compare_groups_via_strata_tool(
    z: ZarrAccess,
    strata: StrataAccess,
    *,
    limit_bytes: int,
) -> Tool:
    async def run(
        obs_column: str,
        group_a: str,
        group_b: str,
        n: int = 20,
    ) -> dict[str, Any]:
        n = min(max(n, 1), HARD_LIMIT)

        # 1. Find a coarse table whose axes == [obs_column]
        matching_slug: str | None = None
        for slug in strata.coarse_slugs():
            if strata.coarse_axes(slug) == [obs_column]:
                matching_slug = slug
                break
        if matching_slug is None:
            return {
                "error": (
                    f"no coarse strata table covers axes [{obs_column!r}] — "
                    f"fall back to compare_groups"
                )
            }

        # 2. Read the coarse table
        coarse = await strata.read_coarse(matching_slug)

        # 3. Find stratum rows for group_a / group_b
        keys_col0 = coarse.stratum_keys[:, 0]
        a_rows = np.where(keys_col0 == group_a)[0]
        b_rows = np.where(keys_col0 == group_b)[0]
        if len(a_rows) == 0:
            return {"error": f"group {group_a!r} not in coarse {matching_slug!r}"}
        if len(b_rows) == 0:
            return {"error": f"group {group_b!r} not in coarse {matching_slug!r}"}
        a_idx, b_idx = int(a_rows[0]), int(b_rows[0])

        # 4. Compute per-gene means + log2-fold-change
        n_a = int(coarse.n_cells[a_idx])
        n_b = int(coarse.n_cells[b_idx])
        if n_a == 0 or n_b == 0:
            return {"error": "at least one group has 0 cells"}
        means_a = coarse.sum_x[a_idx] / n_a
        means_b = coarse.sum_x[b_idx] / n_b
        lfc = np.log2((means_a + PSEUDO) / (means_b + PSEUDO))

        # 5. Top-n by absolute LFC, resolve gene symbols
        names = await z.var_names()
        order = np.argsort(-np.abs(lfc))[:n]
        top_genes = [
            {
                "symbol": str(names[int(i)]),
                "log2_fold_change": float(lfc[i]),
                "mean_a": float(means_a[i]),
                "mean_b": float(means_b[i]),
            }
            for i in order
        ]

        return cap_json_bytes(
            {
                "group_a": group_a,
                "group_b": group_b,
                "method": "log2_fold_change_via_coarse_strata",
                "genes": top_genes,
            },
            limit_bytes=limit_bytes,
        )

    return Tool(
        name="compare_groups_via_strata",
        kind="data",
        description=(
            "Find the top N genes most differentially expressed between two "
            "groups of cells, using precomputed coarse strata tables. Errors if "
            "no coarse table covers the requested obs_column; in that case, "
            "use compare_groups instead. Mirrors compare_groups's argument and "
            "output schema for direct comparison."
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
