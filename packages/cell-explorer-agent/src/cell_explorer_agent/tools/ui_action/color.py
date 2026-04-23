"""set_color_by_gene, set_color_by_category tools."""

from typing import Any

from cell_explorer_agent.schema.app_config import validate_partial
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def set_color_by_gene_tool(z: ZarrAccess) -> Tool:
    async def run(gene: str) -> dict[str, Any]:
        names = await z.var_names()
        if gene not in names:
            return {"error": f"gene {gene!r} not in dataset"}
        payload = {"colorBy": "gene", "gene": gene}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="set_color_by_gene",
        kind="ui_action",
        description="Color the embedding by expression of one gene.",
        args_schema={
            "type": "object",
            "properties": {"gene": {"type": "string"}},
            "required": ["gene"],
            "additionalProperties": False,
        },
        func=run,
    )


def set_color_by_category_tool(z: ZarrAccess) -> Tool:
    async def run(category: str) -> dict[str, Any]:
        cols = await z.obs_columns()
        names = [c.name for c in cols]
        if category not in names:
            return {"error": f"obs column {category!r} not in dataset"}
        payload = {"colorBy": "category", "category": category}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="set_color_by_category",
        kind="ui_action",
        description="Color the embedding by a categorical obs column.",
        args_schema={
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
        func=run,
    )
