"""Protocol for strata-table access. Satisfied by StrataZarrAccess at runtime.

Companion to zarr_protocol.py — keeps strata access separate from the
AnnData-facing ZarrAccess Protocol. v1 exposes coarse-only methods; atomic
methods will be added when the first atomic-consuming tool is designed.
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


@runtime_checkable
class StrataAccess(Protocol):
    """Read-only access to per-stratum aggregate tables (coarse only in v1)."""

    def coarse_slugs(self) -> list[str]: ...

    def coarse_axes(self, slug: str) -> list[str]:
        """Raises KeyError on unknown slug."""
        ...

    async def read_coarse(self, slug: str) -> CoarseStrataResult:
        """Returns the full coarse table for `slug`. Raises KeyError on unknown."""
        ...
