"""StrataStore: companion to AnnDataStore for per-stratum aggregate tables.

Reads uns/strata/coarse_* and uns/strata/atomic groups produced by
`cell2zarr build-strata`. See:
https://github.com/cBioPortal/cell-explorer-py/issues/117
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from zarr_access.zarr_store import ZarrStore


# When a row-selective atomic read needs more than this fraction of the table,
# fall back to a full read + numpy fancy index. Above the threshold, zarr's
# per-chunk byte-range reads add more overhead than the bytes they save (see
# #133 for the empirical data). Below it, chunk-aware selection wins.
ROWS_SLICED_SELECTIVITY_THRESHOLD = 0.5


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

    async def read_atomic_genes(self, gene_indices: list[int]) -> AtomicStrataTable:
        """Read the atomic table restricted to the requested gene columns.

        Empty list → returns a valid table with empty sum_x/sum_xx/nnz but
        populated stratum_keys + n_cells (no gene-data fetches).

        Raises ValueError if the dataset has no atomic table.
        """
        if not self.has_atomic():
            raise ValueError("no atomic table in this dataset")

        group_path = "uns/strata/atomic"
        group = await self._zarr.get_group(group_path)
        schema_version = str(dict(group.attrs).get("schema_version", "1.0"))

        stratum_keys = await self._read_array_2d_str(f"{group_path}/stratum_keys")
        n_cells = await self._read_array_1d(f"{group_path}/n_cells", np.int32)
        axes = self.atomic_axes() or []

        if not gene_indices:
            n_strata = stratum_keys.shape[0]
            return AtomicStrataTable(
                kind="atomic",
                axes=axes,
                stratum_keys=stratum_keys,
                gene_indices=[],
                sum_x=np.zeros((n_strata, 0), dtype=np.float32),
                sum_xx=np.zeros((n_strata, 0), dtype=np.float32),
                nnz=np.zeros((n_strata, 0), dtype=np.int32),
                n_cells=n_cells,
                schema_version=schema_version,
            )

        sum_x = await self._read_array_2d_genes_sliced(
            f"{group_path}/sum_x", gene_indices, np.float32
        )
        sum_xx = await self._read_array_2d_genes_sliced(
            f"{group_path}/sum_xx", gene_indices, np.float32
        )
        nnz = await self._read_array_2d_genes_sliced(
            f"{group_path}/nnz", gene_indices, np.int32
        )

        return AtomicStrataTable(
            kind="atomic",
            axes=axes,
            stratum_keys=stratum_keys,
            gene_indices=list(gene_indices),
            sum_x=sum_x,
            sum_xx=sum_xx,
            nnz=nnz,
            n_cells=n_cells,
            schema_version=schema_version,
        )

    async def read_atomic_stratum_keys(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (stratum_keys, n_cells) for the atomic table without fetching
        the per-gene sum arrays.

        Cheap initial read used by callers that compute which row indices match
        a query value before requesting the row-selective sums.

        Raises ValueError if the dataset has no atomic table.
        """
        if not self.has_atomic():
            raise ValueError("no atomic table in this dataset")

        group_path = "uns/strata/atomic"
        stratum_keys = await self._read_array_2d_str(f"{group_path}/stratum_keys")
        n_cells = await self._read_array_1d(f"{group_path}/n_cells", np.int32)
        return stratum_keys, n_cells

    async def read_atomic_rows(self, row_indices: list[int]) -> AtomicStrataTable:
        """Read the atomic table restricted to the requested stratum rows.

        Per-array reads use zarr fancy indexing: ``arr[row_indices, :]``. Under
        v3 sharding, only the chunks containing the requested rows are fetched
        (via byte-range reads inside each shard). For datasets at the 15M-cell
        scale where the full atomic table is multiple GB, this is what keeps
        per-query data transfer scaling with selectivity instead of total
        table size.

        Empty list returns a valid table with zero rows of gene data but full
        ``stratum_keys`` / ``n_cells`` for inspection.

        Raises ValueError if the dataset has no atomic table.
        """
        if not self.has_atomic():
            raise ValueError("no atomic table in this dataset")

        group_path = "uns/strata/atomic"
        group = await self._zarr.get_group(group_path)
        schema_version = str(dict(group.attrs).get("schema_version", "1.0"))

        stratum_keys = await self._read_array_2d_str(f"{group_path}/stratum_keys")
        n_cells = await self._read_array_1d(f"{group_path}/n_cells", np.int32)
        axes = self.atomic_axes() or []

        sliced_keys = stratum_keys[row_indices] if row_indices else stratum_keys[:0]
        sliced_n_cells = n_cells[row_indices] if row_indices else n_cells[:0]

        if not row_indices:
            n_genes = 0
            sum_x = np.zeros((0, n_genes), dtype=np.float32)
            sum_xx = np.zeros((0, n_genes), dtype=np.float32)
            nnz = np.zeros((0, n_genes), dtype=np.int32)
        else:
            sum_x = await self._read_array_2d_rows_sliced(
                f"{group_path}/sum_x", row_indices, np.float32
            )
            sum_xx = await self._read_array_2d_rows_sliced(
                f"{group_path}/sum_xx", row_indices, np.float32
            )
            nnz = await self._read_array_2d_rows_sliced(
                f"{group_path}/nnz", row_indices, np.int32
            )

        return AtomicStrataTable(
            kind="atomic",
            axes=axes,
            stratum_keys=sliced_keys,
            gene_indices=None,
            sum_x=sum_x,
            sum_xx=sum_xx,
            nnz=nnz,
            n_cells=sliced_n_cells,
            schema_version=schema_version,
        )

    async def read_atomic(self) -> AtomicStrataTable:
        """Read the full atomic table.

        Raises ValueError if the dataset has no atomic table.
        """
        if not self.has_atomic():
            raise ValueError("no atomic table in this dataset")

        group_path = "uns/strata/atomic"
        group = await self._zarr.get_group(group_path)
        schema_version = str(dict(group.attrs).get("schema_version", "1.0"))

        sum_x = await self._read_array_2d(f"{group_path}/sum_x", np.float32)
        sum_xx = await self._read_array_2d(f"{group_path}/sum_xx", np.float32)
        nnz = await self._read_array_2d(f"{group_path}/nnz", np.int32)
        n_cells = await self._read_array_1d(f"{group_path}/n_cells", np.int32)
        stratum_keys = await self._read_array_2d_str(f"{group_path}/stratum_keys")

        return AtomicStrataTable(
            kind="atomic",
            axes=self.atomic_axes() or [],
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

    async def _read_array_2d_genes_sliced(
        self,
        path: str,
        gene_indices: list[int],
        dtype,
    ) -> np.ndarray:
        """Read all rows of an array, restricted to selected gene columns.

        v1 simplification: fetch the whole array then slice in numpy. zarr-python
        supports fancy indexing but this keeps the read path simple for the
        small atomic tables we operate on. Switch to fancy indexing when atlas
        sizes make whole-array fetches expensive.
        """
        arr = await self._zarr.get_array(path)
        data = await arr.getitem((slice(None), slice(None)))
        full = np.asarray(data)
        sliced = full[:, gene_indices]
        return sliced.astype(dtype, copy=False)

    async def _read_array_2d_rows_sliced(
        self,
        path: str,
        row_indices: list[int],
        dtype,
    ) -> np.ndarray:
        """Read a 2D array restricted to selected row indices (all columns).

        Selectivity-aware: above ``ROWS_SLICED_SELECTIVITY_THRESHOLD`` of
        total rows, full read + numpy fancy indexing wins because zarr's
        per-chunk byte-range reads inside each shard add more overhead than
        the bytes they save. Below the threshold, zarr's orthogonal
        selection wins — only the chunks/shards containing the requested
        rows are byte-range fetched. See #133 for the empirical motivation:
        Phase=G1 vs S on Spectrum atomic (~71% selectivity) was 10x slower
        via orthogonal selection than via full read + slice.
        """
        arr = await self._zarr.get_array(path)
        idx = np.asarray(row_indices, dtype=np.int64)
        if len(idx) > arr.shape[0] * ROWS_SLICED_SELECTIVITY_THRESHOLD:
            # Low selectivity: full read + numpy fancy index.
            full = np.asarray(await arr.getitem((slice(None), slice(None))))
            data = full[idx]
        else:
            # High selectivity: chunk-aware orthogonal selection.
            data = await arr.get_orthogonal_selection((idx, slice(None)))
        return np.asarray(data).astype(dtype, copy=False)


def find_coarse_by_axes(strata: StrataStore, axes: list[str]) -> str | None:
    """Find a coarse slug whose axes match the given set exactly (order-insensitive).

    Returns None if no coarse table has exactly these axes.
    """
    target = set(axes)
    for slug in strata.coarse_slugs():
        if set(strata.coarse_axes(slug)) == target:
            return slug
    return None


def find_coarse_covering(strata: StrataStore, required_axes: list[str]) -> str | None:
    """Find the smallest coarse table whose axes are a superset of required_axes.

    Returns None if no coarse covers all the requested axes.
    """
    required = set(required_axes)
    best: tuple[str, int] | None = None  # (slug, axis_count)
    for slug in strata.coarse_slugs():
        slug_axes = set(strata.coarse_axes(slug))
        if not required.issubset(slug_axes):
            continue
        if best is None or len(slug_axes) < best[1]:
            best = (slug, len(slug_axes))
    return best[0] if best else None
