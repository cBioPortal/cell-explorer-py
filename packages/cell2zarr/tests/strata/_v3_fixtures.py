"""Helpers for building v3 AnnData-shaped zarr fixtures.

The strata test suite must mirror what cell2zarr's `convert.py` produces in
production: a zarr v3 store with `X` as a v3 array and `obs` / `var` written
via `anndata.experimental.write_elem`. Using `adata.write_zarr(path)` defaults
to zarr v2, which diverges silently from production and masks bugs in the
strata code path.

This helper is the test-side equivalent of the relevant subset of `convert.py`:
explicit `zarr_format=3` root, manual `X` array, `write_elem` for obs/var,
`consolidate_metadata` at the end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from anndata.io import write_elem


def write_v3_anndata_zarr(
    path: Path,
    X: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
) -> Path:
    """Build a v3 AnnData-shaped zarr fixture at `path`.

    Mirrors cell2zarr's `convert.py` production layout: a v3 root with the
    anndata encoding attrs, a manually-created `X` array, obs/var written via
    `anndata.experimental.write_elem`, and consolidated metadata.

    Returns the same `path` for convenience in fixture chains.
    """
    n_obs, n_vars = X.shape
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["encoding-type"] = "anndata"
    root.attrs["encoding-version"] = "0.1.0"

    X_arr = root.create_array(
        "X",
        shape=X.shape,
        chunks=X.shape,
        dtype=X.dtype,
    )
    X_arr[:] = X
    X_arr.attrs["encoding-type"] = "array"
    X_arr.attrs["encoding-version"] = "0.2.0"

    write_elem(root, "obs", obs)
    write_elem(root, "var", var)

    zarr.consolidate_metadata(str(path), zarr_format=3)
    return path
