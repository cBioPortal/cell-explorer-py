"""set_viewport, clear_viewport, fit_viewport_to_selection tools."""

from typing import Any

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool


def clear_viewport_tool() -> Tool:
    async def run() -> dict[str, Any]:
        # `viewport: null` is the frontend's reset sentinel: applyConfig calls
        # setViewport(null), which bumps viewportEpoch and triggers a fit-to-view
        # via deck.gl's setProps (computed from embedding bounds + container).
        payload: dict[str, Any] = {"viewport": None}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="clear_viewport",
        kind="ui_action",
        description=(
            "Reset the viewport to fit-to-view (the default zoom-out that "
            "shows the entire embedding). Use when the user says 'reset the "
            "view' / 'zoom out' / 'show everything'."
        ),
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        func=run,
    )


def fit_viewport_to_selection_tool() -> Tool:
    async def run() -> dict[str, Any]:
        # Frontend computes the bbox of currently selected cells (from
        # selectionFilterBuffer × embeddingData.positions) and applies a
        # viewport that fits them. No-op if no selection is active.
        payload: dict[str, Any] = {"fitViewportToSelection": True}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="fit_viewport_to_selection",
        kind="ui_action",
        description=(
            "Zoom in on the currently selected/filtered cells. Computes the "
            "bounding box of cells matching the active filter or selection and "
            "applies a viewport that fits them. Use after filter_by_ids / "
            "filter_by_gene_expression when the user says 'zoom in on the "
            "selection' / 'fit to my filter' / 'focus on these cells'. No-op "
            "if no selection is active."
        ),
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        func=run,
    )


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
