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


# Above this many distinct values a column is not a usable filter, and shipping
# its values would dominate the catalogue response. Cardinality is still
# reported — it is free from the consolidated metadata.
FACET_VALUE_CAP = 100


@dataclass(frozen=True)
class ObsFacet:
    """One obs column's facet-relevant facts, keyed by its real column name."""

    dtype: str                # "categorical" | "numeric" | "string"
    cardinality: int | None   # None unless categorical with a readable size
    values: list[str] | None  # None unless categorical and under the cap


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
    obs_facets: dict[str, "ObsFacet"]


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


def _categories_key(zarr_version: int, column: str) -> str:
    """Consolidated-metadata key for a column's categories node."""
    base = f"obs/{column}/categories"
    return base if zarr_version == 3 else f"{base}/.zarray"


def _column_attrs_key(zarr_version: int, column: str) -> str:
    return f"obs/{column}" if zarr_version == 3 else f"obs/{column}/.zattrs"


def _node_shape(node: dict | None) -> list | None:
    """`shape` sits at the node's top level in both v2 and v3 documents."""
    if not isinstance(node, dict):
        return None
    shape = node.get("shape")
    if shape is None:
        attrs = node.get("attributes")
        shape = attrs.get("shape") if isinstance(attrs, dict) else None
    return shape if isinstance(shape, list) else None


def _discover_obs_facets(zarr_store: ZarrStore, obs_columns: list[str]) -> dict[str, ObsFacet]:
    """Facet facts for every obs column, from metadata alone — no requests.

    Cardinality comes from the categories node's shape, so the cap can be
    applied before any read: a 927,205-category column costs nothing to skip.
    """
    cm = zarr_store.consolidated_metadata or {}
    version = zarr_store.zarr_version
    facets: dict[str, ObsFacet] = {}

    for column in obs_columns:
        cat_node = cm.get(_categories_key(version, column))
        if cat_node is not None:
            shape = _node_shape(cat_node)
            # An unreadable shape means unknown size, which is treated as over
            # the cap — an unknown is exactly what the cap guards against.
            cardinality = int(shape[0]) if shape else None
            facets[column] = ObsFacet("categorical", cardinality, None)
            continue

        node = cm.get(_column_attrs_key(version, column)) or {}
        raw = node.get("data_type") or node.get("dtype") or ""
        numeric = str(raw).lstrip("|<>").startswith(("i", "u", "f", "b"))
        facets[column] = ObsFacet("numeric" if numeric else "string", None, None)

    return facets


def _eligible_for_values(column: str, facet: ObsFacet) -> bool:
    return (
        facet.dtype == "categorical"
        and facet.cardinality is not None
        and facet.cardinality <= FACET_VALUE_CAP
        # Ontology ids duplicate a readable column and are never useful labels.
        and not column.endswith("_ontology_term_id")
    )


async def _read_categories(zarr_store: ZarrStore, column: str) -> list[str]:
    """Read one column's category labels. Small: cardinality is already capped."""
    arr = await zarr_store.get_array(f"obs/{column}/categories")
    return [str(v) for v in await arr.getitem(slice(None))]


async def _harvest_obs_facets(
    zarr_store: ZarrStore, obs_columns: list[str]
) -> dict[str, ObsFacet]:
    facets = _discover_obs_facets(zarr_store, obs_columns)
    for column, facet in list(facets.items()):
        if not _eligible_for_values(column, facet):
            continue
        try:
            values = await _read_categories(zarr_store, column)
        except Exception:
            # One unreadable column costs that column, not the dataset.
            continue
        facets[column] = ObsFacet(facet.dtype, facet.cardinality, values)
    return facets


async def extract_store_metadata(zarr_store: ZarrStore) -> StoreMetadata:
    """Read catalog-facing metadata from an open store.

    Raises whatever AnnDataStore.open raises for a store that is not a
    readable AnnData store.
    """
    anndata = await AnnDataStore.open(zarr_store)
    x_encoding, x_dtype = await _x_encoding_and_dtype(zarr_store)
    obs_columns = list(anndata.obs_columns)
    obs_facets = await _harvest_obs_facets(zarr_store, obs_columns)

    return StoreMetadata(
        n_obs=anndata.n_obs,
        n_vars=anndata.n_vars,
        zarr_version=zarr_store.zarr_version,
        obsm_keys=list(anndata.obsm_keys),
        obs_columns=obs_columns,
        var_columns=list(anndata.var_columns),
        layers=await _layer_keys(zarr_store),
        x_dtype=x_dtype,
        x_encoding=x_encoding,
        obs_facets=obs_facets,
    )
