"""Extract catalog-facing facts from an open zarr store.

AnnDataStore.open() already derives n_obs, n_vars, obsm_keys, obs_columns and
var_columns from attrs alone — when consolidated metadata is present those reads
are served from the cached document, so this costs no extra round trip. Wrapping
it (rather than re-parsing the consolidated dict) guarantees the catalog's
numbers match what chat reports for the same dataset.

Deliberately strict: AnnDataStore.open() raises for a non-AnnData store or an
absent X, and this does not soften that. Catalog datasets are cell2zarr output
or curated, so they have a proper encoding-type and an X. Callers convert the
exception into a recorded harvest error.
"""

from dataclasses import dataclass
from typing import Literal

from zarr_access import AnnDataStore, ZarrStore


@dataclass(frozen=True)
class StoreMetadata:
    """Facts about a zarr store, all readable without touching cell data."""

    n_obs: int
    n_vars: int
    zarr_version: Literal[2, 3]
    obsm_keys: list[str]
    obs_columns: list[str]
    var_columns: list[str]
    layers: list[str]
    x_dtype: str | None
    x_encoding: str | None  # "array" | "csr_matrix" | "csc_matrix"


async def _layer_keys(zarr_store: ZarrStore) -> list[str]:
    """Array names under `layers`, or [] when the group is absent."""
    try:
        group = await zarr_store.get_group("layers")
    except Exception:
        return []
    return [key async for key in group.array_keys()]


async def _x_encoding_and_dtype(zarr_store: ZarrStore) -> tuple[str | None, str | None]:
    """Describe X without reading any of it.

    Mirrors AnnDataStore.open's group-then-array probe: sparse X is a group
    carrying `encoding-type`, dense X is a plain array.
    """
    try:
        x_group = await zarr_store.get_group("X")
        encoding = dict(x_group.attrs).get("encoding-type")
        # get_group() does not itself verify node type — it happily returns a
        # dense X array to us and reads its attrs. Only treat this as the
        # sparse case when encoding-type actually says so, otherwise fall
        # through to the dense-array probe below, mirroring AnnDataStore.open.
        if encoding not in ("csr_matrix", "csc_matrix"):
            raise ValueError(f"X is not a sparse group (encoding-type={encoding!r})")
        try:
            data = await zarr_store.get_array("X/data")
            return encoding, str(data.dtype)
        except Exception:
            return encoding, None
    except Exception:
        pass

    try:
        x_arr = await zarr_store.get_array("X")
        return "array", str(x_arr.dtype)
    except Exception:
        return None, None


async def extract_store_metadata(zarr_store: ZarrStore) -> StoreMetadata:
    """Read catalog-facing metadata from an open store.

    Raises whatever AnnDataStore.open raises for a store that is not a
    readable AnnData store.
    """
    anndata = await AnnDataStore.open(zarr_store)
    x_encoding, x_dtype = await _x_encoding_and_dtype(zarr_store)

    return StoreMetadata(
        n_obs=anndata.n_obs,
        n_vars=anndata.n_vars,
        zarr_version=zarr_store.zarr_version,
        obsm_keys=list(anndata.obsm_keys),
        obs_columns=list(anndata.obs_columns),
        var_columns=list(anndata.var_columns),
        layers=await _layer_keys(zarr_store),
        x_dtype=x_dtype,
        x_encoding=x_encoding,
    )
