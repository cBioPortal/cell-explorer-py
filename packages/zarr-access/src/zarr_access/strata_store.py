"""StrataStore: companion to AnnDataStore for per-stratum aggregate tables.

Reads uns/strata/coarse_* and uns/strata/atomic groups produced by
`cell2zarr build-strata`. See:
https://github.com/cBioPortal/cell-explorer-py/issues/117
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from zarr_access.zarr_store import ZarrStore


@dataclass(frozen=True)
class CoarseStrataTable:
    """A coarse strata table — sums aggregated over a subset of atomic axes."""

    kind: Literal["coarse"]
    slug: str
    axes: list[str]
    stratum_keys: np.ndarray         # (n_strata, n_axes) string
    gene_indices: list[int] | None   # None for whole-table reads
    sum_x: np.ndarray                # (n_strata, n_genes) float32
    sum_xx: np.ndarray
    nnz: np.ndarray                  # int32
    n_cells: np.ndarray              # (n_strata,) int32
    schema_version: str


@dataclass(frozen=True)
class AtomicStrataTable:
    """The atomic strata table — sums per (cell_type × donor × ...) tuple."""

    kind: Literal["atomic"]
    axes: list[str]
    stratum_keys: np.ndarray
    gene_indices: list[int] | None
    sum_x: np.ndarray
    sum_xx: np.ndarray
    nnz: np.ndarray
    n_cells: np.ndarray
    schema_version: str


StrataTable = CoarseStrataTable | AtomicStrataTable


class StrataStore:
    """Companion to AnnDataStore that reads uns/strata/* groups."""

    def __init__(self, zarr_store: ZarrStore) -> None:
        self._zarr = zarr_store

    @classmethod
    async def open(cls, zarr_store: ZarrStore) -> "StrataStore":
        """Open a StrataStore against a ZarrStore. Discovery happens here."""
        return cls(zarr_store)
