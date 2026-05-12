"""add_summary_obs_column, add_summary_gene, and their removal counterparts."""

from typing import Any

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def add_summary_obs_column_tool(z: ZarrAccess) -> Tool:
    async def run(obs_column: str) -> dict[str, Any]:
        cols = await z.obs_columns()
        if obs_column not in {c.name for c in cols}:
            return {"error": f"obs column {obs_column!r} not in dataset"}
        payload = {"summaryObsColumns": [obs_column]}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="add_summary_obs_column",
        kind="ui_action",
        description="Add an obs column to the viewer's summary panel.",
        args_schema={
            "type": "object",
            "properties": {"obs_column": {"type": "string"}},
            "required": ["obs_column"],
            "additionalProperties": False,
        },
        func=run,
    )


def add_summary_gene_tool(z: ZarrAccess) -> Tool:
    async def run(gene: str) -> dict[str, Any]:
        names = await z.var_names()
        if gene not in names:
            return {"error": f"gene {gene!r} not in dataset"}
        payload = {"summaryGenes": [gene]}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="add_summary_gene",
        kind="ui_action",
        description="Add a gene to the viewer's summary panel.",
        args_schema={
            "type": "object",
            "properties": {"gene": {"type": "string"}},
            "required": ["gene"],
            "additionalProperties": False,
        },
        func=run,
    )


def remove_summary_obs_column_tool() -> Tool:
    async def run(obs_column: str) -> dict[str, Any]:
        # No dataset validation — the frontend silently no-ops if the column
        # isn't pinned, which matches the "remove if present" intent. The
        # agent might also be asked to remove something the user pinned via
        # the sidebar without the agent knowing it.
        payload: dict[str, Any] = {"removeSummaryObsColumns": [obs_column]}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="remove_summary_obs_column",
        kind="ui_action",
        description=(
            "Unpin one obs column from the summary panel. Inverse of "
            "add_summary_obs_column. Silently no-ops if the column isn't "
            "currently pinned."
        ),
        args_schema={
            "type": "object",
            "properties": {"obs_column": {"type": "string"}},
            "required": ["obs_column"],
            "additionalProperties": False,
        },
        func=run,
    )


def remove_summary_gene_tool() -> Tool:
    async def run(gene: str) -> dict[str, Any]:
        # Same rationale as remove_summary_obs_column — no dataset validation.
        # The frontend resolves symbol → var index and silently no-ops on miss.
        payload: dict[str, Any] = {"removeSummaryGenes": [gene]}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="remove_summary_gene",
        kind="ui_action",
        description=(
            "Unpin one gene from the summary panel. Inverse of "
            "add_summary_gene. Silently no-ops if the gene isn't currently "
            "pinned."
        ),
        args_schema={
            "type": "object",
            "properties": {"gene": {"type": "string"}},
            "required": ["gene"],
            "additionalProperties": False,
        },
        func=run,
    )


def clear_summary_tool() -> Tool:
    async def run() -> dict[str, Any]:
        # Composite payload: both null sentinels clear both pinned lists in
        # one applyConfig call.
        payload: dict[str, Any] = {"summaryObsColumns": None, "summaryGenes": None}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="clear_summary",
        kind="ui_action",
        description=(
            "Unpin every obs column and gene from the summary panel. Use when "
            "the user says 'clear the summary' / 'remove everything from "
            "summary' / 'reset summary'."
        ),
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        func=run,
    )
