"""set_gene_label_column ui_action tool."""

from typing import Any

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def set_gene_label_column_tool(z: ZarrAccess) -> Tool:
    async def run(column: str) -> dict[str, Any]:
        cols = await z.var_columns()
        if column not in cols:
            return {"error": f"var column {column!r} not in dataset (available: {cols})"}
        payload = {"geneLabelColumn": column}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="set_gene_label_column",
        kind="ui_action",
        description=(
            "Set the var dataframe column used to display gene labels (e.g. "
            "'feature_id' or 'gene_symbol'). Different datasets use different "
            "naming conventions; this lets you switch the displayed labels "
            "between e.g. Ensembl IDs and HGNC symbols."
        ),
        args_schema={
            "type": "object",
            "properties": {"column": {"type": "string"}},
            "required": ["column"],
            "additionalProperties": False,
        },
        func=run,
    )
