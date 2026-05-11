"""filter_by_gene_expression — filter cells by gene expression range."""

from typing import Any

import numpy as np

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def filter_by_gene_expression_tool(z: ZarrAccess) -> Tool:
    async def run(
        gene: str,
        min: float | None = None,
        max: float | None = None,
    ) -> dict[str, Any]:
        if min is None and max is None:
            return {"error": "specify at least one of min or max"}
        if min is not None and max is not None and min > max:
            return {"error": f"min ({min}) > max ({max})"}

        names = await z.var_names()
        if gene not in names:
            return {"error": f"gene {gene!r} not in dataset"}

        # Pre-fetch expression to verify the filter matches some cells and to
        # report the match count back to the LLM for narration. One column
        # read — same cost as gene_expression_summary.
        expr = await z.gene_column(gene)
        lo = min if min is not None else float("-inf")
        hi = max if max is not None else float("inf")
        mask = (expr >= lo) & (expr <= hi)
        count = int(np.count_nonzero(mask))
        if count == 0:
            return {
                "error": f"no cells match {gene} in [{min}, {max}]"
            }

        payload = {
            "filterByExpression": {"gene": gene, "min": min, "max": max}
        }
        validate_partial(payload)
        return {"payload": payload, "matched_cells": count}

    return Tool(
        name="filter_by_gene_expression",
        kind="ui_action",
        description=(
            "Filter the viewer to cells whose `gene` expression is in [min, max]. "
            "Either bound is optional — pass only `min` for ≥ threshold, only `max` "
            "for ≤ threshold, or both for a range. Use for 'show me high-CD8A "
            "cells' / 'cells with CD3D > 2' / 'isolate low-MS4A1 cells' style "
            "queries. Call gene_expression_summary first to learn the expression "
            "range before picking thresholds — the values are not always 0–1."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "gene": {"type": "string"},
                "min": {"type": ["number", "null"]},
                "max": {"type": ["number", "null"]},
            },
            "required": ["gene"],
            "additionalProperties": False,
        },
        func=run,
    )
