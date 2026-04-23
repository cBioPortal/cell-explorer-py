"""Dataset context builder — metadata block for the system prompt."""

from dataclasses import dataclass
from typing import Literal

from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


@dataclass
class ObsColumnInfo:
    name: str
    dtype: Literal["categorical", "numeric", "string"]
    cardinality: int | None


@dataclass
class DatasetContext:
    slug: str
    name: str
    description: str
    n_obs: int
    n_var: int
    obs_columns: list[ObsColumnInfo]
    embedding_keys: list[str]


async def build_dataset_context(
    z: ZarrAccess, *, slug: str, name: str, description: str
) -> DatasetContext:
    n_obs, n_var = await z.shape()
    obs = await z.obs_columns()
    emb = await z.obsm_keys()
    return DatasetContext(
        slug=slug,
        name=name,
        description=description,
        n_obs=n_obs,
        n_var=n_var,
        obs_columns=[
            ObsColumnInfo(name=c.name, dtype=c.dtype, cardinality=c.cardinality)
            for c in obs
        ],
        embedding_keys=list(emb),
    )
