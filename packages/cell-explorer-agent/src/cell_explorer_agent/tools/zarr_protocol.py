"""Protocol for zarr dataset access. Satisfied by zarr-access at runtime."""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np


@dataclass
class ObsColumnSpec:
    name: str
    dtype: Literal["categorical", "numeric", "string"]
    cardinality: int | None = None  # for categorical
    categories: list[str] | None = None  # populated for low-cardinality categoricals


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
    async def var_columns(self) -> list[str]:
        """Names of columns in the var dataframe (e.g., 'feature_id', 'gene_symbol').

        Distinct from var_names which is the gene-name index.
        """
        ...

    async def var_column(self, name: str) -> ObsColumn:
        """Single var column — efficient for targeted queries.

        Returns the same ObsColumn shape as obs_column (categorical/numeric/string
        with values + categories). The type name reflects historical naming;
        the structure is axis-agnostic.
        """
        ...
    async def obsm_keys(self) -> list[str]: ...
    async def gene_index(self, gene: str) -> int: ...
    async def gene_column(self, gene: str) -> np.ndarray: ...
    async def obs_mask(self, obs_col: str, value: str) -> np.ndarray: ...
