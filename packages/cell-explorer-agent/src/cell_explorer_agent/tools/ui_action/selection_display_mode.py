"""set_selection_display_mode ui_action tool."""

from typing import Any, Literal

from cell_explorer_agent.schema import validate_partial, ValidationError
from cell_explorer_agent.tools.registry import Tool


def set_selection_display_mode_tool() -> Tool:
    async def run(value: Literal["dim", "hide"]) -> dict[str, Any]:
        payload = {"selectionDisplayMode": value}
        try:
            validate_partial(payload)
        except ValidationError:
            return {"error": f"invalid selectionDisplayMode value: {value!r}"}
        return {"payload": payload}

    return Tool(
        name="set_selection_display_mode",
        kind="ui_action",
        description=(
            "Switch how non-selected cells render when a filter is active: 'dim' "
            "keeps them visible at low opacity, 'hide' culls them entirely from "
            "the canvas. After filter_by_ids / filter_by_gene_expression the "
            "default is 'hide'; call this with 'dim' when the user asks to keep "
            "the rest of the canvas visible (e.g. 'show me T cells in context' "
            "or 'filter to high-CD8A but keep the others faded')."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "value": {"type": "string", "enum": ["dim", "hide"]},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        func=run,
    )
