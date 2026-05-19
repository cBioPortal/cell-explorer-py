"""Protocol for strata-table access. Satisfied by StrataZarrAccess at runtime.

Companion to zarr_protocol.py — keeps strata access separate from the
AnnData-facing ZarrAccess Protocol. Exposes both coarse (single-axis)
tables and the multi-axis atomic table; callers can aggregate the
atomic table on-the-fly when no coarse covers their query axis.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CoarseStrataResult:
    """Wire-format view of a coarse strata table for tool consumption.

    Mirrors zarr_access.CoarseStrataTable's shape but lives in the agent
    package so tools don't import zarr_access directly. The
    StrataZarrAccess adapter converts between the two.
    """

    axes: list[str]
    stratum_keys: np.ndarray         # (n_strata, n_axes) string
    sum_x: np.ndarray                # (n_strata, n_genes) float32
    sum_xx: np.ndarray
    nnz: np.ndarray
    n_cells: np.ndarray              # (n_strata,) int32


@dataclass(frozen=True)
class AtomicStrataResult:
    """Wire-format view of the atomic strata table.

    Same row shape as CoarseStrataResult but with potentially many axes
    (cell_type × donor × ... × supersite). Callers aggregate rows by an
    axis value to derive single-axis sums on the fly.
    """

    axes: list[str]
    stratum_keys: np.ndarray         # (n_strata, n_axes) string
    sum_x: np.ndarray                # (n_strata, n_genes) float32
    sum_xx: np.ndarray
    nnz: np.ndarray
    n_cells: np.ndarray              # (n_strata,) int32


@runtime_checkable
class StrataAccess(Protocol):
    """Read-only access to per-stratum aggregate tables."""

    def coarse_slugs(self) -> list[str]: ...

    def coarse_axes(self, slug: str) -> list[str]:
        """Raises KeyError on unknown slug."""
        ...

    async def read_coarse(self, slug: str) -> CoarseStrataResult:
        """Returns the full coarse table for `slug`. Raises KeyError on unknown."""
        ...

    def has_atomic(self) -> bool: ...

    def atomic_axes(self) -> list[str] | None:
        """Axes covered by the atomic table, or None if the dataset has none."""
        ...

    async def read_atomic(self) -> AtomicStrataResult:
        """Returns the full atomic table. Raises ValueError if no atomic table."""
        ...

    async def read_atomic_stratum_keys(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (stratum_keys, n_cells) for the atomic table without fetching
        the per-gene sum arrays. Cheap initial read used by callers that compute
        which row indices match a query value before requesting the row-selective
        sums. Raises ValueError if no atomic table.
        """
        ...

    async def read_atomic_rows(self, row_indices: list[int]) -> AtomicStrataResult:
        """Returns the atomic table restricted to the given stratum rows (all
        genes). Under v3 sharding with stratum-major chunks, only the chunks
        containing the requested rows are fetched. Raises ValueError if no
        atomic table.
        """
        ...
