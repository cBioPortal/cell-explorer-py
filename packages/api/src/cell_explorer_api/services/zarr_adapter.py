"""Adapter from zarr_access.AnnDataStore to cell_explorer_agent's ZarrAccess Protocol."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cell_explorer_agent.tools.zarr_protocol import ObsColumn, ObsColumnSpec


@dataclass
class AnnDataZarrAccess:
    """Wraps a zarr_access.AnnDataStore to satisfy the ZarrAccess Protocol.

    Pure translation — no business logic. Caches per-session where it helps.
    """

    store: Any  # zarr_access.AnnDataStore; Any to avoid importing here
    _obs_columns_cache: list[ObsColumnSpec] | None = field(default=None, init=False)
    _var_names_cache: list[str] | None = field(default=None, init=False)

    async def shape(self) -> tuple[int, int]:
        return (self.store.n_obs, self.store.n_vars)

    async def attrs(self) -> dict:
        # Not exposed by AnnDataStore; no v1 tool reads attrs.
        return {}

    async def obsm_keys(self) -> list[str]:
        return list(self.store.obsm_keys)

    async def var_names(self) -> list[str]:
        if self._var_names_cache is None:
            var_df = await self.store.var()
            self._var_names_cache = list(var_df.index)
        return list(self._var_names_cache)

    async def obs_column(self, name: str) -> ObsColumn:
        raw = await self.store.obs_column(name)
        return _decode_col_to_obs_column(name, raw)

    async def gene_index(self, gene: str) -> int:
        var_df = await self.store.var()
        return var_df.index.get_loc(gene)

    async def gene_column(self, gene: str) -> np.ndarray:
        return await self.store.gene_expression(gene)

    async def obs_mask(self, obs_col: str, value: str) -> np.ndarray:
        col = await self.obs_column(obs_col)
        if col.dtype != "categorical" or col.categories is None:
            raise ValueError(
                f"obs_mask requires a categorical column; {obs_col!r} is {col.dtype}"
            )
        code = col.categories.index(value)
        return col.values == code

    async def obs_columns(self) -> list[ObsColumnSpec]:
        if self._obs_columns_cache is None:
            specs: list[ObsColumnSpec] = []
            for name in self.store.obs_columns:
                raw = await self.store.obs_column(name)
                col = _decode_col_to_obs_column(name, raw)
                specs.append(ObsColumnSpec(
                    name=col.name,
                    dtype=col.dtype,
                    cardinality=len(col.categories) if col.categories else None,
                ))
            self._obs_columns_cache = specs
        return list(self._obs_columns_cache)


def _decode_col_to_obs_column(name: str, data) -> ObsColumn:
    """Translate zarr_access.decode_column output into ObsColumn.

    decode_column returns `np.ndarray | pd.Categorical`:
    - `pd.Categorical` → categorical, with integer codes + string category labels.
    - numeric `np.ndarray` → numeric.
    - anything else `np.ndarray` (object/string) → string.
    """
    if isinstance(data, pd.Categorical):
        return ObsColumn(
            name=name,
            dtype="categorical",
            values=np.asarray(data.codes),
            categories=[str(c) for c in data.categories],
        )
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.number):
        return ObsColumn(
            name=name,
            dtype="numeric",
            values=arr.astype("float64"),
            categories=None,
        )
    return ObsColumn(
        name=name,
        dtype="string",
        values=arr.astype(object),
        categories=None,
    )
