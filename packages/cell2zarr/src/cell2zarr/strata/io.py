"""Zarr read + write helpers for cell2zarr.strata.

Read primitives operate on a `zarr.Group` (the AnnData zarr root).
Write primitives (write_atomic, write_coarse) use a completion-marker pattern:
schema_version is written LAST so partial writes are detectable.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import zarr

if TYPE_CHECKING:
    from cell2zarr.strata.build import AtomicTable
    from cell2zarr.strata.derive import CoarseTable

# AnnData encodes a categorical obs column as a group with two arrays:
#   <col>/codes      int32  cell -> category index
#   <col>/categories <T>    one entry per category (string or int)
# This matches AnnData's "categorical" encoding-type. We don't depend on
# anndata at runtime here; we read the codes + categories directly.


def open_dataset(path: Path) -> zarr.Group:
    """Open an AnnData zarr store for read/write.

    use_consolidated=False bypasses the consolidated metadata cache written by
    AnnData so that new groups added at uns/strata/ are immediately visible
    without rewriting the consolidated metadata file.

    Raises FileNotFoundError if the path doesn't exist.
    """
    return zarr.open_group(str(path), mode="r+", use_consolidated=False)


def read_obs_categorical(root: zarr.Group, column: str) -> pd.Categorical:
    """Read an obs categorical column from an AnnData zarr group.

    Raises KeyError if the column is missing under obs/.
    """
    if "obs" not in root:
        raise KeyError("no obs/ group in zarr root")
    obs = root["obs"]
    if column not in obs:
        raise KeyError(f"obs column '{column}' not found")

    col = obs[column]
    if "codes" in col and "categories" in col:
        codes = np.asarray(col["codes"])
        categories = np.asarray(col["categories"])
        return pd.Categorical.from_codes(codes=codes, categories=categories.tolist())

    # Plain (non-categorical) column — wrap as Categorical for uniform handling
    values = np.asarray(col)
    return pd.Categorical(values)


def read_x_block(root: zarr.Group, gene_slice: slice) -> np.ndarray:
    """Read a contiguous gene-column block from X as float32.

    Returns array of shape (n_obs, k) where k = gene_slice.stop - gene_slice.start.
    Upcasts to float32 regardless of underlying dtype.
    """
    if "X" not in root:
        raise KeyError("no X/ array in zarr root")
    X = root["X"]
    block = np.asarray(X[:, gene_slice])
    return block.astype(np.float32, copy=False)


def has_strata(root: zarr.Group) -> bool:
    """True iff a *fully-written* uns/strata/atomic/ exists (schema_version present)."""
    try:
        atomic = root["uns"]["strata"]["atomic"]
    except KeyError:
        return False
    return "schema_version" in atomic.attrs


def existing_strata_summary(root: zarr.Group) -> dict | None:
    """Return {axes, n_strata, schema_version} for an existing atomic table, or None.

    Partial / mid-build writes (no schema_version yet) return None.
    """
    try:
        atomic = root["uns"]["strata"]["atomic"]
    except KeyError:
        return None
    if "schema_version" not in atomic.attrs:
        return None
    return {
        "schema_version": atomic.attrs["schema_version"],
        "axes": list(atomic.attrs.get("axes", [])),
        "n_strata": int(atomic.attrs.get("n_strata", 0)),
    }


# ---------------------------------------------------------------------------
# Write primitives
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
import zarr.codecs  # noqa: E402

ATOMIC_ROWS_PER_CHUNK = 5
ATOMIC_ROWS_PER_SHARD = 50

from cell2zarr.strata.exceptions import StrataExistsError  # noqa: E402


def _strata_group(root: zarr.Group) -> zarr.Group:
    """Get-or-create uns/strata/ on the AnnData zarr root."""
    return root.require_group("uns").require_group("strata")


def _write_arrays_with_marker(
    parent: zarr.Group,
    name: str,
    arrays: dict[str, np.ndarray],
    attrs: dict,
    *,
    array_options: dict[str, dict] | None = None,
) -> None:
    """Atomicity helper: write all arrays, then attach the schema_version marker LAST.

    A reader that finds the group without schema_version treats it as a
    partial / aborted write.

    Per-array zarr `create_array` kwargs (chunks, shards, compressors, etc.)
    can be supplied via `array_options`, keyed by array name. Arrays without
    an entry use zarr's defaults.
    """
    # Drop any existing group (called only when we're already authorized to overwrite).
    if name in parent:
        del parent[name]
    group = parent.create_group(name)

    options = array_options or {}
    for array_name, data in arrays.items():
        kwargs = options.get(array_name, {})
        group.create_array(array_name, data=data, **kwargs)

    # Attach non-marker attrs first.
    for key, value in attrs.items():
        if key == "schema_version":
            continue
        group.attrs[key] = value
    # Marker LAST — the "this group is complete" flag.
    group.attrs["schema_version"] = SCHEMA_VERSION


def _atomic_array_options(n_strata: int, n_genes: int) -> dict[str, dict]:
    """Build the per-array zarr options for the atomic table's large arrays.

    Uses stratum-major chunks bundled into medium shards so future row-selective
    reads can fetch only the strata a query needs. Clamps to `n_strata` for
    small datasets and ensures shard is an integer multiple of chunk.
    """
    chunk_rows = min(ATOMIC_ROWS_PER_CHUNK, n_strata)
    shard_rows = min(ATOMIC_ROWS_PER_SHARD, n_strata)
    shard_rows = (shard_rows // chunk_rows) * chunk_rows or chunk_rows
    compressors = [zarr.codecs.ZstdCodec(level=3)]
    return {
        name: {
            "chunks": (chunk_rows, n_genes),
            "shards": (shard_rows, n_genes),
            "compressors": compressors,
        }
        for name in ("sum_x", "sum_xx", "nnz")
    }


def write_atomic(root: zarr.Group, atomic: AtomicTable, *, force: bool) -> None:
    """Write an AtomicTable to uns/strata/atomic/.

    Raises StrataExistsError if a finished atomic table is already present and
    force is False. Partial / husk groups (no schema_version) are silently
    overwritten regardless of force.
    """
    strata = _strata_group(root)

    if has_strata(root) and not force:
        existing_axes = list(strata["atomic"].attrs.get("axes", []))
        raise StrataExistsError(
            f"uns/strata/atomic already exists (axes={existing_axes}). "
            f"Pass --force to overwrite."
        )

    n_strata, n_genes = atomic.sum_x.shape
    array_options = _atomic_array_options(n_strata, n_genes)
    _write_arrays_with_marker(
        parent=strata,
        name="atomic",
        arrays={
            "stratum_keys": atomic.stratum_keys,
            "sum_x": atomic.sum_x,
            "sum_xx": atomic.sum_xx,
            "nnz": atomic.nnz,
            "n_cells": atomic.n_cells,
        },
        attrs={
            "axes": atomic.axis_names,
            "n_strata": int(len(atomic.stratum_keys)),
            "derived_from": None,
            "collapsed_axes": [],
        },
        array_options=array_options,
    )


def read_atomic(root: zarr.Group) -> AtomicTable:
    """Load an already-written AtomicTable back from zarr.

    Raises KeyError if uns/strata/atomic/ is missing or incomplete (no
    schema_version marker).
    """
    # Lazy import to avoid build.py ↔ io.py circular at module load time.
    from cell2zarr.strata.build import AtomicTable

    if not has_strata(root):
        raise KeyError("no completed uns/strata/atomic/ found in dataset")
    atomic = root["uns"]["strata"]["atomic"]
    return AtomicTable(
        axis_names=list(atomic.attrs["axes"]),
        stratum_keys=np.asarray(atomic["stratum_keys"]),
        sum_x=np.asarray(atomic["sum_x"]),
        sum_xx=np.asarray(atomic["sum_xx"]),
        nnz=np.asarray(atomic["nnz"]),
        n_cells=np.asarray(atomic["n_cells"]),
    )


def consolidate_strata_metadata(root: zarr.Group) -> None:
    """Re-write the consolidated metadata index so newly-added uns/strata/* groups
    are visible to readers that open with `use_consolidated=True` (the default).

    On zarr v3 stores, consolidated metadata is stored *inline* in the root
    `zarr.json` under the `consolidated_metadata` key (not in a separate
    `.zmetadata` file as in v2). It's a snapshot of every child node's
    metadata, so any additions after the snapshot was taken are invisible to
    readers that trust it. Calling `zarr.consolidate_metadata(store)` rewrites
    the snapshot to include whatever we just added.
    """
    zarr.consolidate_metadata(root.store)


def write_coarse(
    root: zarr.Group,
    name: str,
    coarse: CoarseTable,
    *,
    force: bool,
) -> None:
    """Write a CoarseTable to uns/strata/coarse_<name>/.

    `name` is used directly as the group name slug (the caller decides the
    slug, typically by joining axis names with underscores).
    """
    strata = _strata_group(root)
    group_name = f"coarse_{name}"

    if (
        group_name in strata
        and "schema_version" in strata[group_name].attrs
        and not force
    ):
        raise StrataExistsError(
            f"uns/strata/{group_name} already exists. Pass --force to overwrite."
        )

    _write_arrays_with_marker(
        parent=strata,
        name=group_name,
        arrays={
            "stratum_keys": coarse.stratum_keys,
            "sum_x": coarse.sum_x,
            "sum_xx": coarse.sum_xx,
            "nnz": coarse.nnz,
            "n_cells": coarse.n_cells,
        },
        attrs={
            "axes": coarse.axis_names,
            "n_strata": int(len(coarse.stratum_keys)),
            "derived_from": "atomic",
            "collapsed_axes": [],
        },
    )
