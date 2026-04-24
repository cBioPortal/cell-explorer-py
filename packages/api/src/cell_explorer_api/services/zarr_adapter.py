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
