"""set_category_labels ui_action tool."""

from typing import Any

from cell_explorer_agent.schema import validate_partial, ValidationError
from cell_explorer_agent.tools.registry import Tool


def set_category_labels_tool() -> Tool:
    async def run(value: bool, obs_column: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"showCategoryLabels": value}
        if obs_column is not None:
            payload["categoryLabelsObsColumn"] = obs_column
        try:
            validate_partial(payload)
        except ValidationError:
            return {
                "error": f"invalid set_category_labels payload: value={value!r} obs_column={obs_column!r}"
            }
        return {"payload": payload}

    return Tool(
        name="set_category_labels",
        kind="ui_action",
        description=(
            "Toggle the cluster-label overlay on the scatterplot. The "
            "optional obs_column arg picks which categorical obs column "
            "to label by; without it, labels follow the current color "
            "obs column when in category color mode (or render nothing "
            "in gene mode). When ON and a column is resolved, the "
            "category names render at each cluster's centroid. Labels "
            "are highlight-aware only when labeling by the same column "
            "being colored — when labeling by a different column, "
            "highlights don't apply. Use when the user asks "
            "'show cluster labels', 'label the clusters by leiden', "
            "'turn off labels', etc."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "value": {"type": "boolean"},
                "obs_column": {"type": ["string", "null"]},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        func=run,
    )
