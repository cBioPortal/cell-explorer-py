"""Validation for a dataset's stored default_view.

Reuses the generated AppConfig Pydantic model (from cell-explorer-agent, itself
codegen'd from the frontend zod schema via `make generate-app-config`) for field
and type validation, then constrains to the v1 subset and enforces the cross-field
rules that the frontend's applyConfig.ts also enforces.
"""

from cell_explorer_agent.schema.app_config import Model as AppConfig
from pydantic import ValidationError

# Fields the highperformer viewer applies in v1. Widen this set as the frontend
# learns to apply more of the AppConfig (embedding, viewport, filters, chrome).
V1_ALLOWED_KEYS = {
    "colorBy",
    "gene",
    "category",
    "colorScaleName",
    "highlightedCategories",
    "showCategoryLabels",
    "categoryLabelsObsColumn",
    "pointSize",
    "opacity",
}


class DefaultViewError(ValueError):
    """Raised when a default_view payload is invalid."""


def validate_default_view(value: dict) -> dict:
    """Validate a default_view dict, returning it unchanged if valid.

    Raises DefaultViewError on any failure.
    """
    # Layer 1: field/type validation via the generated AppConfig model.
    # extra='forbid' rejects keys unknown to the frontend schema; enum/bounds
    # (colorScaleName, colorBy, pointSize > 0, 0 <= opacity <= 1) come for free.
    try:
        AppConfig.model_validate(value)
    except ValidationError as exc:
        raise DefaultViewError(str(exc)) from exc

    # Layer 2: v1 allowed-keys guard — reject real AppConfig fields not yet applied.
    extra = set(value) - V1_ALLOWED_KEYS
    if extra:
        raise DefaultViewError(
            f"unsupported field(s) for v1 default_view: {sorted(extra)}"
        )

    # Layer 3: cross-field rules (mirror packages/highperformer/src/config/applyConfig.ts).
    color_by = value.get("colorBy")
    if color_by == "gene" and not value.get("gene"):
        raise DefaultViewError("colorBy='gene' requires 'gene'")
    if color_by == "category" and not value.get("category"):
        raise DefaultViewError("colorBy='category' requires 'category'")
    if "highlightedCategories" in value and color_by != "category":
        raise DefaultViewError("highlightedCategories requires colorBy='category'")

    return value
