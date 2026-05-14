"""filter_by_ids, clear_filter tools."""

from typing import Any

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def filter_by_ids_tool(z: ZarrAccess, *, filter_ids_max: int) -> Tool:
    async def run(obs_column: str, ids: list[str]) -> dict[str, Any]:
        if len(ids) > filter_ids_max:
            return {
                "error": f"id list ({len(ids)}) exceeds cap {filter_ids_max}"
            }
        cols = await z.obs_columns()
        if obs_column not in {c.name for c in cols}:
            return {"error": f"obs column {obs_column!r} not in dataset"}
        payload = {"filter": {"obsColumn": obs_column, "ids": list(ids)}}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="filter_by_ids",
        kind="ui_action",
        description=(
            "Filter the viewer to a subset of cells identified by ids within an "
            "obs column. Resolve the id list before calling — no free-form queries."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "obs_column": {"type": "string"},
                "ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["obs_column", "ids"],
            "additionalProperties": False,
        },
        func=run,
    )


def clear_filter_tool() -> Tool:
    async def run() -> dict[str, Any]:
        payload = {"filter": {"obsColumn": "_none", "ids": []}}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="clear_filter",
        kind="ui_action",
        description=(
            "Clear any active filter in the viewer (filter_by_ids or "
            "filter_by_gene_expression). Call this for 'clear filter', "
            "'undo filter', 'remove filter', 'show all cells', or 'show "
            "everything again'."
        ),
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        func=run,
    )
