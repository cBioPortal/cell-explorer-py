"""set_embedding tool."""

from typing import Any

from cell_explorer_agent.schema import validate_partial
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def set_embedding_tool(z: ZarrAccess) -> Tool:
    async def run(embedding: str) -> dict[str, Any]:
        keys = await z.obsm_keys()
        if embedding not in keys:
            return {"error": f"embedding {embedding!r} not available; have {keys}"}
        payload = {"embedding": embedding}
        validate_partial(payload)
        return {"payload": payload}

    return Tool(
        name="set_embedding",
        kind="ui_action",
        description="Switch the active embedding (obsm key) in the viewer.",
        args_schema={
            "type": "object",
            "properties": {"embedding": {"type": "string"}},
            "required": ["embedding"],
            "additionalProperties": False,
        },
        func=run,
    )
