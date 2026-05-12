"""set_color_by_gene, set_color_by_category, set_color_scale, clear_color_by tools."""

from typing import Any, Literal

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def set_color_scale_tool() -> Tool:
    async def run(
        name: Literal["viridis", "magma", "plasma", "inferno"],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"colorScaleName": name}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="set_color_scale",
        kind="ui_action",
        description=(
            "Switch the continuous color scale used for gene-mode coloring. "
            "Affects how expression values map to RGB; the categorical palette "
            "is independent. Use when the user asks for a specific palette "
            "('switch to plasma' / 'use inferno') or expresses a perceptual "
            "preference. Default is viridis."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["viridis", "magma", "plasma", "inferno"],
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        func=run,
    )


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


def clear_color_by_tool() -> Tool:
    async def run() -> dict[str, Any]:
        # `colorBy: null` is the frontend's reset sentinel: applyConfig calls
        # clearGene + clearObsColumn, and the canvas falls back to default gray.
        payload: dict[str, Any] = {"colorBy": None}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="clear_color_by",
        kind="ui_action",
        description=(
            "Reset the embedding coloring to the default (gray points). Use "
            "when the user says 'stop coloring' / 'clear the color' / 'reset "
            "color'. Does not affect filters or rendering controls."
        ),
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        func=run,
    )


def set_color_by_category_tool(z: ZarrAccess) -> Tool:
    async def run(
        category: str,
        highlight: list[str] | None = None,
    ) -> dict[str, Any]:
        cols = await z.obs_columns()
        names = [c.name for c in cols]
        if category not in names:
            return {"error": f"obs column {category!r} not in dataset"}

        payload: dict[str, Any] = {"colorBy": "category", "category": category}

        if highlight is not None:
            col = await z.obs_column(category)
            if col.dtype != "categorical" or col.categories is None:
                return {
                    "error": f"obs column {category!r} is not categorical; cannot highlight values"
                }
            allowed = set(col.categories)
            unknown = [v for v in highlight if v not in allowed]
            if unknown:
                return {
                    "error": (
                        f"value(s) {unknown!r} not in obs column {category!r}; "
                        f"available: {col.categories}"
                    )
                }
            payload["highlightedCategories"] = list(highlight)

        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="set_color_by_category",
        kind="ui_action",
        description=(
            "Color the embedding by a categorical obs column. Optionally pass "
            "`highlight` — a list of category values that should render at full "
            "opacity while the rest fade to gray. Use highlight when the user "
            "asks 'show me the T cells' / 'where are the B cells': pick the obs "
            "column for `category` and pass the requested value(s) in `highlight`."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "highlight": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["category"],
            "additionalProperties": False,
        },
        func=run,
    )
