"""Opened AnnData zarr stores, cached across tool calls by dataset slug."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import zarr

from zarr_access import AnnDataStore, StrataStore, ZarrStore
from zarr_access.strata_store import CoarseStrataTable

# Var columns that hold human-readable gene symbols, in priority order. Matches
# the API's chat adapter and the frontend, so all three name genes the same way
# (e.g. SPECTRUM indexes var by Ensembl ID and keeps symbols in feature_name).
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
    "gene",
)

DEFAULT_MAX_DATASETS = 8


@dataclass
class OpenDataset:
    """One opened dataset plus lazily-populated per-dataset caches."""

    slug: str
    url: str
    zarr: ZarrStore
    anndata: AnnDataStore
    strata: StrataStore
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _symbols: list[str] | None = field(default=None, init=False)
    _var_ids: list[str] | None = field(default=None, init=False)
    _symbol_column: str | None = field(default=None, init=False)
    _gene_lookup: dict[str, int] | None = field(default=None, init=False)
    _obs_cache: dict[str, np.ndarray | pd.Categorical] = field(default_factory=dict, init=False)
    _coarse_cache: dict[str, CoarseStrataTable] = field(default_factory=dict, init=False)

    @property
    def n_obs(self) -> int:
        return self.anndata.n_obs

    @property
    def n_vars(self) -> int:
        return self.anndata.n_vars

    async def _ensure_genes(self) -> None:
        async with self._lock:
            if self._gene_lookup is not None:
                return
            var_df = await self.anndata.var()
            symbol_column = next((c for c in GENE_SYMBOL_COLUMNS if c in var_df.columns), None)
            ids = [str(v) for v in var_df.index]
            symbols = [str(v) for v in var_df[symbol_column].values] if symbol_column else ids
            # Exact symbol, then exact ID, then case-insensitive forms; the
            # first gene registered under a key wins.
            lookup: dict[str, int] = {}
            for names in (symbols, ids):
                for i, name in enumerate(names):
                    lookup.setdefault(name, i)
            for names in (symbols, ids):
                for i, name in enumerate(names):
                    lookup.setdefault(name.upper(), i)
            self._symbols = symbols
            self._var_ids = ids
            self._symbol_column = symbol_column
            self._gene_lookup = lookup

    async def gene_symbols(self) -> list[str]:
        await self._ensure_genes()
        assert self._symbols is not None
        return self._symbols

    async def var_ids(self) -> list[str]:
        await self._ensure_genes()
        assert self._var_ids is not None
        return self._var_ids

    async def symbol_column(self) -> str | None:
        await self._ensure_genes()
        return self._symbol_column

    async def gene_index(self, gene: str) -> int:
        """Resolve a gene symbol or var ID (case-insensitive) to its column in X."""
        await self._ensure_genes()
        assert self._gene_lookup is not None
        idx = self._gene_lookup.get(gene)
        if idx is None:
            idx = self._gene_lookup.get(gene.upper())
        if idx is None:
            raise KeyError(gene)
        return idx

    async def gene_column(self, gene_idx: int) -> np.ndarray:
        """Expression of one gene across all cells, without loading all of X."""
        x_node = await self.zarr.get_array("X")
        if isinstance(x_node, zarr.AsyncArray):
            data = await x_node.getitem((slice(None), gene_idx))
            return np.asarray(data).astype("float32", copy=False).ravel()
        # Sparse X: AnnDataStore reads the whole matrix. Every published
        # dataset is dense, so this only serves small local stores.
        ids = await self.var_ids()
        return await self.anndata.gene_expression(ids[gene_idx])

    async def dense_x(self) -> "zarr.AsyncArray | None":
        x_node = await self.zarr.get_array("X")
        return x_node if isinstance(x_node, zarr.AsyncArray) else None

    async def obs_column(self, name: str) -> np.ndarray | pd.Categorical:
        if name not in self.anndata.obs_columns:
            raise KeyError(name)
        if name not in self._obs_cache:
            self._obs_cache[name] = await self.anndata.obs_column(name)
        return self._obs_cache[name]

    def coarse_slug_for(self, obs_column: str) -> str | None:
        for slug in self.strata.coarse_slugs():
            if self.strata.coarse_axes(slug) == [obs_column]:
                return slug
        return None

    async def coarse_table(self, slug: str) -> CoarseStrataTable:
        if slug not in self._coarse_cache:
            self._coarse_cache[slug] = await self.strata.read_coarse(slug)
        return self._coarse_cache[slug]


async def open_dataset(slug: str, url: str) -> OpenDataset:
    zarr_store = await ZarrStore.open(url)
    anndata = await AnnDataStore.open(zarr_store)
    strata = await StrataStore.open(zarr_store)
    return OpenDataset(slug=slug, url=url, zarr=zarr_store, anndata=anndata, strata=strata)


class DatasetCache:
    """LRU of OpenDataset keyed by slug.

    An entry whose URL no longer matches the catalog (dataset re-ingested to a
    new location) is reopened. Concurrent requests for the same slug share one
    open.
    """

    def __init__(self, maxsize: int = DEFAULT_MAX_DATASETS, opener=open_dataset) -> None:
        self._maxsize = maxsize
        self._opener = opener
        self._entries: OrderedDict[str, OpenDataset] = OrderedDict()
        self._opening: dict[str, asyncio.Lock] = {}

    async def get(self, slug: str, url: str) -> OpenDataset:
        entry = self._entries.get(slug)
        if entry is not None and entry.url == url:
            self._entries.move_to_end(slug)
            return entry

        lock = self._opening.setdefault(slug, asyncio.Lock())
        async with lock:
            entry = self._entries.get(slug)
            if entry is not None and entry.url == url:
                self._entries.move_to_end(slug)
                return entry
            entry = await self._opener(slug, url)
            self._entries[slug] = entry
            self._entries.move_to_end(slug)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
            return entry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, slug: str) -> bool:
        return slug in self._entries
