"""set_summary_context ui_action tool."""

from typing import Any, Literal

from cell_explorer_agent.schema import validate_partial, ValidationError
from cell_explorer_agent.tools.registry import Tool


def set_summary_context_tool() -> Tool:
    async def run(value: Literal["overall", "selections"]) -> dict[str, Any]:
        payload = {"summaryContext": value}
        try:
            validate_partial(payload)
        except ValidationError:
            return {"error": f"invalid summaryContext value: {value!r}"}
        return {"payload": payload}

    return Tool(
        name="set_summary_context",
        kind="ui_action",
        description=(
            "Switch the summary panel between 'overall' (statistics across all "
            "cells in the dataset) and 'selections' (statistics restricted to "
            "the current selection/filter). Useful when you want to compare a "
            "selection's stats against the dataset average."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "value": {"type": "string", "enum": ["overall", "selections"]},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        func=run,
    )
