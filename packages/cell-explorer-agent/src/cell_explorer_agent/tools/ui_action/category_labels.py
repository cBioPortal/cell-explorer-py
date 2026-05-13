"""set_category_labels ui_action tool."""

from typing import Any

from cell_explorer_agent.schema import validate_partial, ValidationError
from cell_explorer_agent.tools.registry import Tool


def set_category_labels_tool() -> Tool:
    async def run(value: bool) -> dict[str, Any]:
        payload = {"showCategoryLabels": value}
        try:
            validate_partial(payload)
        except ValidationError:
            return {"error": f"invalid showCategoryLabels value: {value!r}"}
        return {"payload": payload}

    return Tool(
        name="set_category_labels",
        kind="ui_action",
        description=(
            "Toggle the cluster-label overlay on the scatterplot. When ON and the "
            "user is coloring by a categorical obs column, the category names "
            "(e.g. cell types) render at each cluster's centroid. Labels are "
            "highlight-aware — if categories are highlighted, only those labels "
            "render. Has no visible effect outside category color mode; the "
            "toggle is persisted regardless so it takes effect once the user "
            "switches back. Use when the user asks 'show cluster labels', "
            "'label the clusters', 'turn off labels', etc."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "value": {"type": "boolean"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        func=run,
    )
