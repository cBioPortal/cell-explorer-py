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

    def __init__(
        self,
        zarr_store: ZarrStore,
        *,
        atomic_axes: list[str] | None,
        atomic_strata_count: int | None,
        coarse_axes_map: dict[str, list[str]],
        coarse_strata_count_map: dict[str, int],
    ) -> None:
        self._zarr = zarr_store
        self._atomic_axes = atomic_axes
        self._atomic_strata_count = atomic_strata_count
        self._coarse_axes_map = coarse_axes_map
        self._coarse_strata_count_map = coarse_strata_count_map

    @classmethod
    async def open(cls, zarr_store: ZarrStore) -> "StrataStore":
        """Open a StrataStore. Discovery happens here — single async pass over uns/strata/."""
        atomic_axes: list[str] | None = None
        atomic_strata_count: int | None = None
        coarse_axes_map: dict[str, list[str]] = {}
        coarse_strata_count_map: dict[str, int] = {}

        try:
            strata_group = await zarr_store.get_group("uns/strata")
        except (KeyError, FileNotFoundError):
            return cls(
                zarr_store,
                atomic_axes=None,
                atomic_strata_count=None,
                coarse_axes_map={},
                coarse_strata_count_map={},
            )

        # Atomic group
        try:
            atomic_group = await strata_group.getitem("atomic")
            attrs = dict(atomic_group.attrs)
            if attrs.get("schema_version") is not None:
                axes = attrs.get("axes")
                if isinstance(axes, list):
                    atomic_axes = list(axes)
                n_strata = attrs.get("n_strata")
                if isinstance(n_strata, int):
                    atomic_strata_count = n_strata
        except (KeyError, FileNotFoundError):
            pass

        # Coarse_* siblings — enumerate group children
        async for child_name in strata_group.group_keys():
            if not child_name.startswith("coarse_"):
                continue
            slug = child_name[len("coarse_"):]
            try:
                coarse_group = await strata_group.getitem(child_name)
                attrs = dict(coarse_group.attrs)
                if attrs.get("schema_version") is None:
                    continue
                axes = attrs.get("axes")
                if isinstance(axes, list):
                    coarse_axes_map[slug] = list(axes)
                n_strata = attrs.get("n_strata")
                if isinstance(n_strata, int):
                    coarse_strata_count_map[slug] = n_strata
            except (KeyError, FileNotFoundError):
                continue

        return cls(
            zarr_store,
            atomic_axes=atomic_axes,
            atomic_strata_count=atomic_strata_count,
            coarse_axes_map=coarse_axes_map,
            coarse_strata_count_map=coarse_strata_count_map,
        )

    # --- Discovery (sync) ---

    def has_atomic(self) -> bool:
        return self._atomic_axes is not None

    def atomic_axes(self) -> list[str] | None:
        return list(self._atomic_axes) if self._atomic_axes else None

    def atomic_strata_count(self) -> int | None:
        return self._atomic_strata_count

    def coarse_slugs(self) -> list[str]:
        return sorted(self._coarse_axes_map.keys())

    def coarse_axes(self, slug: str) -> list[str]:
        if slug not in self._coarse_axes_map:
            raise KeyError(slug)
        return list(self._coarse_axes_map[slug])

    def coarse_strata_count(self, slug: str) -> int:
        if slug not in self._coarse_strata_count_map:
            raise KeyError(slug)
        return self._coarse_strata_count_map[slug]
