"""set_viewport ui_action tool — pan and zoom the scatterplot."""

from typing import Any

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool


def set_viewport_tool() -> Tool:
    async def run(target_x: float, target_y: float, zoom: float) -> dict[str, Any]:
        payload = {"viewport": {"target": [target_x, target_y], "zoom": zoom}}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="set_viewport",
        kind="ui_action",
        description=(
            "Pan and zoom the scatterplot. target_x and target_y are coordinates "
            "in the embedding's space (NOT pixel coordinates); zoom is log-scale "
            "(0 ≈ fit-to-view, 1 ≈ 2x, 2 ≈ 4x, negative values zoom out). The "
            "frontend captures these into shareable URLs when the user clicks "
            "Share."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "target_x": {"type": "number"},
                "target_y": {"type": "number"},
                "zoom": {"type": "number"},
            },
            "required": ["target_x", "target_y", "zoom"],
            "additionalProperties": False,
        },
        func=run,
    )
