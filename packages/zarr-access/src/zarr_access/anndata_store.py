"""High-level AnnData API on top of ZarrStore."""

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
