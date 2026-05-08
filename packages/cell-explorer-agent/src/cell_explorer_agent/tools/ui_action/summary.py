"""add_summary_obs_column, add_summary_gene tools."""

from typing import Any

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def add_summary_obs_column_tool(z: ZarrAccess) -> Tool:
    async def run(obs_column: str) -> dict[str, Any]:
        cols = await z.obs_columns()
        if obs_column not in {c.name for c in cols}:
            return {"error": f"obs column {obs_column!r} not in dataset"}
        payload = {"summaryObsColumns": [obs_column]}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="add_summary_obs_column",
        kind="ui_action",
        description="Add an obs column to the viewer's summary panel.",
        args_schema={
            "type": "object",
            "properties": {"obs_column": {"type": "string"}},
            "required": ["obs_column"],
            "additionalProperties": False,
        },
        func=run,
    )


def add_summary_gene_tool(z: ZarrAccess) -> Tool:
    async def run(gene: str) -> dict[str, Any]:
        names = await z.var_names()
        if gene not in names:
            return {"error": f"gene {gene!r} not in dataset"}
        payload = {"summaryGenes": [gene]}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="add_summary_gene",
        kind="ui_action",
        description="Add a gene to the viewer's summary panel.",
        args_schema={
            "type": "object",
            "properties": {"gene": {"type": "string"}},
            "required": ["gene"],
            "additionalProperties": False,
        },
        func=run,
    )
