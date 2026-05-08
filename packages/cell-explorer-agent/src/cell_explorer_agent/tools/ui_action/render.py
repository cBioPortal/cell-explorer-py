"""set_render_controls ui_action tool — point size and opacity."""

from typing import Any

from cell_explorer_agent.schema import validate_partial, ValidationError
from cell_explorer_agent.tools.registry import Tool


def set_render_controls_tool() -> Tool:
    async def run(
        point_size: float | None = None,
        opacity: float | None = None,
    ) -> dict[str, Any]:
        if point_size is None and opacity is None:
            return {"error": "specify at least one of point_size or opacity"}
        payload: dict[str, Any] = {}
        if point_size is not None:
            payload["pointSize"] = point_size
        if opacity is not None:
            payload["opacity"] = opacity
        try:
            validate_partial(payload)
        except ValidationError as exc:
            return {"error": f"invalid render controls: {exc.errors()[0]['msg']}"}
        return {"payload": payload}

    return Tool(
        name="set_render_controls",
        kind="ui_action",
        description=(
            "Adjust scatterplot rendering. point_size in pixels (must be > 0; "
            "default ~2). opacity from 0 (transparent) to 1 (opaque). At least "
            "one parameter must be specified."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "point_size": {"type": "number", "exclusiveMinimum": 0},
                "opacity": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "additionalProperties": False,
        },
        func=run,
    )
