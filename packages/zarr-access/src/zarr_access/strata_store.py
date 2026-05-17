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

    # --- Reads ---

    async def read_coarse(self, slug: str) -> CoarseStrataTable:
        """Read the full coarse table for `slug`.

        Raises KeyError synchronously if the slug is unknown.
        """
        if slug not in self._coarse_axes_map:
            raise KeyError(slug)

        group_path = f"uns/strata/coarse_{slug}"
        group = await self._zarr.get_group(group_path)
        schema_version = str(dict(group.attrs).get("schema_version", "1.0"))

        sum_x = await self._read_array_2d(f"{group_path}/sum_x", np.float32)
        sum_xx = await self._read_array_2d(f"{group_path}/sum_xx", np.float32)
        nnz = await self._read_array_2d(f"{group_path}/nnz", np.int32)
        n_cells = await self._read_array_1d(f"{group_path}/n_cells", np.int32)
        stratum_keys = await self._read_array_2d_str(f"{group_path}/stratum_keys")

        return CoarseStrataTable(
            kind="coarse",
            slug=slug,
            axes=self.coarse_axes(slug),
            stratum_keys=stratum_keys,
            gene_indices=None,
            sum_x=sum_x,
            sum_xx=sum_xx,
            nnz=nnz,
            n_cells=n_cells,
            schema_version=schema_version,
        )

    # --- Read primitives ---

    async def _read_array_2d(self, path: str, dtype) -> np.ndarray:
        arr = await self._zarr.get_array(path)
        data = await arr.getitem((slice(None), slice(None)))
        return np.asarray(data).astype(dtype, copy=False)

    async def _read_array_1d(self, path: str, dtype) -> np.ndarray:
        arr = await self._zarr.get_array(path)
        data = await arr.getitem(slice(None))
        return np.asarray(data).astype(dtype, copy=False)

    async def _read_array_2d_str(self, path: str) -> np.ndarray:
        arr = await self._zarr.get_array(path)
        data = await arr.getitem((slice(None), slice(None)))
        return np.asarray(data)
