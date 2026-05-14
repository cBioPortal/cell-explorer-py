"""Zarr read + write helpers for cell2zarr.strata.

Read primitives operate on a `zarr.Group` (the AnnData zarr root).
Writes are added in a later task.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

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
