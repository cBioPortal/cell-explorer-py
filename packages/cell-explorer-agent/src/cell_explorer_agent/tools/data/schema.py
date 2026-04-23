"""get_dataset_schema — returns shape, obs/var metadata, and embeddings."""

from typing import Any

from cell_explorer_agent.tools.caps import cap_json_bytes
from cell_explorer_agent.tools.registry import Tool
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def get_dataset_schema_tool(z: ZarrAccess, *, limit_bytes: int) -> Tool:
    async def run() -> dict[str, Any]:
        n_obs, n_var = await z.shape()
        cols = await z.obs_columns()
        obsm = await z.obsm_keys()
        payload = {
            "n_obs": n_obs,
            "n_var": n_var,
            "var_count": n_var,
            "obs_columns": [
                {"name": c.name, "dtype": c.dtype, "cardinality": c.cardinality}
                for c in cols
            ],
            "embeddings": list(obsm),
        }
        return cap_json_bytes(payload, limit_bytes=limit_bytes)

    return Tool(
        name="get_dataset_schema",
        kind="data",
        description=(
            "Return dataset shape, obs column names/dtypes/cardinalities, "
            "and available embedding keys. Call this first to learn the schema."
        ),
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        func=run,
    )
