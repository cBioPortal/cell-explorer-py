"""High-level AnnData API on top of ZarrStore."""

import numpy as np
import pandas as pd
import scipy.sparse

from zarr_access.decoders import (
    decode_column,
    decode_dataframe,
    decode_sparse_matrix,
)
from zarr_access.zarr_store import ZarrStore


class AnnDataStore:
    """AnnData API layered on top of ZarrStore (async)."""

    def __init__(
        self,
        zarr_store: ZarrStore,
        n_obs: int,
        n_vars: int,
        obsm_keys: list[str],
        obs_columns: list[str],
        var_columns: list[str],
    ):
        self._zarr = zarr_store
        self._n_obs = n_obs
        self._n_vars = n_vars
        self._obsm_keys = obsm_keys
        self._obs_columns = obs_columns
        self._var_columns = var_columns

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def obsm_keys(self) -> list[str]:
        return list(self._obsm_keys)

    @property
    def obs_columns(self) -> list[str]:
        return list(self._obs_columns)

    @property
    def var_columns(self) -> list[str]:
        return list(self._var_columns)

    async def obs(self) -> pd.DataFrame:
        """Full obs dataframe."""
        group = await self._zarr.get_group("obs")
        return await decode_dataframe(group)

    async def var(self) -> pd.DataFrame:
        """Full var dataframe."""
        group = await self._zarr.get_group("var")
        return await decode_dataframe(group)

    async def obs_column(self, name: str):
        """Single obs column — efficient for targeted queries."""
        obs = await self._zarr.get_group("obs")
        node = await obs.getitem(name)
        return await decode_column(node)

    async def obsm(self, key: str) -> np.ndarray:
        """Embedding array (e.g., X_umap)."""
        obsm = await self._zarr.get_group("obsm")
        arr = await obsm.getitem(key)
        data = await arr.getitem(slice(None))
        return np.asarray(data)

    async def gene_expression(self, gene: str) -> np.ndarray:
        """Expression values for a single gene across all cells.

        Resolves gene name to column index via var index, then reads that
        column from X.
        """
        var_df = await self.var()
        if gene not in var_df.index:
            raise ValueError(f"Gene {gene!r} not found in var index")
        col_idx = var_df.index.get_loc(gene)
        x = await self.X()
        if scipy.sparse.issparse(x):
            return np.asarray(x[:, col_idx].toarray()).ravel()
        return np.asarray(x[:, col_idx]).ravel()

    async def X(self):
        """Full expression matrix (dense or sparse depending on encoding)."""
        # Try as a group first (sparse case)
        try:
            x_group = await self._zarr.get_group("X")
            attrs = dict(x_group.attrs)
            if attrs.get("encoding-type") in ("csr_matrix", "csc_matrix"):
                return await decode_sparse_matrix(x_group)
        except (KeyError, TypeError, ValueError):
            pass
        # Dense array
        x_arr = await self._zarr.get_array("X")
        data = await x_arr.getitem((slice(None), slice(None)))
        return np.asarray(data)

    @classmethod
    async def open(cls, zarr_store: ZarrStore) -> "AnnDataStore":
        """Open an AnnData zarr store, validating encoding-type."""
        root_attrs = dict(zarr_store._root.attrs)
        if root_attrs.get("encoding-type") != "anndata":
            raise ValueError(
                f"Expected encoding-type 'anndata', got {root_attrs.get('encoding-type')!r}"
            )

        # Resolve shape by checking X — it may be a dense array or a sparse group
        try:
            x_group = await zarr_store.get_group("X")
            # X is a group — check if it's a sparse matrix
            x_attrs = dict(x_group.attrs)
            enc = x_attrs.get("encoding-type")
            if enc in ("csr_matrix", "csc_matrix"):
                shape = x_attrs.get("shape")
                if not shape:
                    raise ValueError("Sparse X missing 'shape' attribute")
                n_obs, n_vars = int(shape[0]), int(shape[1])
            else:
                raise ValueError(f"Unexpected X encoding: {enc}")
        except Exception:
            # Not a group — try as a dense array
            x_arr = await zarr_store.get_array("X")
            n_obs, n_vars = int(x_arr.shape[0]), int(x_arr.shape[1])

        # Discover obsm keys — obsm arrays are sub-arrays inside obsm group
        obsm_keys: list[str] = []
        try:
            obsm_group = await zarr_store.get_group("obsm")
            async for key in obsm_group.array_keys():
                obsm_keys.append(key)
        except Exception:
            pass

        # Get obs and var column order from dataframe attrs
        obs_columns: list[str] = []
        var_columns: list[str] = []
        try:
            obs_group = await zarr_store.get_group("obs")
            obs_columns = list(dict(obs_group.attrs).get("column-order", []))
        except Exception:
            pass
        try:
            var_group = await zarr_store.get_group("var")
            var_columns = list(dict(var_group.attrs).get("column-order", []))
        except Exception:
            pass

        return cls(zarr_store, n_obs, n_vars, obsm_keys, obs_columns, var_columns)
