"""Protocol for zarr dataset access. Satisfied by zarr-access at runtime."""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np


@dataclass
class ObsColumnSpec:
    name: str
    dtype: Literal["categorical", "numeric", "string"]
    cardinality: int | None = None  # for categorical


@dataclass
class ObsColumn:
    name: str
    dtype: Literal["categorical", "numeric", "string"]
    values: np.ndarray  # numeric vector OR int codes for categorical
    categories: list[str] | None = None  # only for categorical
    index: np.ndarray | None = None  # obs names, aligned to values


@runtime_checkable
class ZarrAccess(Protocol):
    async def shape(self) -> tuple[int, int]: ...
    async def attrs(self) -> dict: ...
    async def obs_columns(self) -> list[ObsColumnSpec]: ...
    async def obs_column(self, name: str) -> ObsColumn: ...
    async def var_names(self) -> list[str]: ...
    async def obsm_keys(self) -> list[str]: ...
    async def gene_index(self, gene: str) -> int: ...
    async def gene_column(self, gene: str) -> np.ndarray: ...
    async def obs_mask(self, obs_col: str, value: str) -> np.ndarray: ...
