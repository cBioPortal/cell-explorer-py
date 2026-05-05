"""Adapter from zarr_access.AnnDataStore to cell_explorer_agent's ZarrAccess Protocol."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cell_explorer_agent.tools.zarr_protocol import ObsColumn, ObsColumnSpec


# Candidate var columns (in priority order) that hold human-readable gene symbols.
# Mirrors the frontend's GENE_SYMBOL_COLUMNS so the CLI and browser agree on how
# genes are named. The first column present in a given dataset wins.
GENE_SYMBOL_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "gene_symbol",
    "GeneSymbol",
    "gene_symbols",
    "var_name",
    "gene_name",
    "gene_short_name",
    "symbol",
    "name",
)


@dataclass
class AnnDataZarrAccess:
    """Wraps a zarr_access.AnnDataStore to satisfy the ZarrAccess Protocol.

    Pure translation — no business logic. Caches per-session where it helps.

    Gene name resolution prefers a well-known symbol column (see
    GENE_SYMBOL_COLUMNS) over the raw var index, because datasets like MSK
    SPECTRUM use Ensembl IDs as the index and store human symbols in
    `feature_name`. Falls back to the index if none of those columns exist.

    Gene column reads use zarr column-wise slicing directly on the X array to
    avoid AnnDataStore.gene_expression's full-matrix load, which would OOM on
    any production-sized dataset.
    """

    store: Any  # zarr_access.AnnDataStore; Any to avoid importing here
    _obs_columns_cache: list[ObsColumnSpec] | None = field(default=None, init=False)
    _var_names_cache: list[str] | None = field(default=None, init=False)
    _gene_col_name_cache: str | None = field(default=None, init=False)
    _gene_to_row_cache: dict[str, int] | None = field(default=None, init=False)

    async def shape(self) -> tuple[int, int]:
        return (self.store.n_obs, self.store.n_vars)

    async def attrs(self) -> dict:
        # Not exposed by AnnDataStore; no v1 tool reads attrs.
        return {}

    async def obsm_keys(self) -> list[str]:
        return list(self.store.obsm_keys)

    async def _ensure_gene_map(self) -> None:
        """Build the gene-name → row-index map, using a symbol column if present."""
        if self._gene_to_row_cache is not None:
            return
        var_df = await self.store.var()
        gene_col = None
        for candidate in GENE_SYMBOL_COLUMNS:
            if candidate in var_df.columns:
                gene_col = candidate
                break
        if gene_col is not None:
            names = [str(v) for v in var_df[gene_col].values]
        else:
            names = [str(v) for v in var_df.index]
        # For duplicate names (rare but possible), the first occurrence wins.
        mapping: dict[str, int] = {}
        for i, name in enumerate(names):
            mapping.setdefault(name, i)
        self._gene_col_name_cache = gene_col  # None if using index
        self._var_names_cache = names
        self._gene_to_row_cache = mapping

    async def var_names(self) -> list[str]:
        await self._ensure_gene_map()
        assert self._var_names_cache is not None
        return list(self._var_names_cache)

    async def obs_column(self, name: str) -> ObsColumn:
        raw = await self.store.obs_column(name)
        return _decode_col_to_obs_column(name, raw)

    async def gene_index(self, gene: str) -> int:
        await self._ensure_gene_map()
        assert self._gene_to_row_cache is not None
        if gene not in self._gene_to_row_cache:
            raise KeyError(gene)
        return self._gene_to_row_cache[gene]

    async def gene_column(self, gene: str) -> np.ndarray:
        """Read one column of X without loading the whole matrix."""
        col_idx = await self.gene_index(gene)
        zarr_store = self.store._zarr  # zarr_access.ZarrStore (private on purpose)

        # Try as a dense zarr array first; column-wise slice is cheap.
        try:
            x_arr = await zarr_store.get_array("X")
        except Exception:
            x_arr = None

        if x_arr is not None:
            data = await x_arr.getitem((slice(None), col_idx))
            return np.asarray(data).astype("float32", copy=False).ravel()

        # Sparse fallback: delegate to AnnDataStore.gene_expression. For large
        # sparse matrices this still pays a non-trivial cost; we accept that
        # for v1 since Spectrum (and similar float16 dense stores) won't hit it.
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
                    categories=list(col.categories) if col.categories else None,
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
