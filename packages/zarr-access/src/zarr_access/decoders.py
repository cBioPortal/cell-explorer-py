"""Decoders for AnnData encoding conventions."""

import numpy as np
import pandas as pd
import scipy.sparse
import zarr


async def _read_array(arr: zarr.AsyncArray) -> np.ndarray:
    """Read an async zarr array into numpy."""
    data = await arr.getitem(slice(None))
    return np.asarray(data)


async def _read_string_array(arr: zarr.AsyncArray) -> np.ndarray:
    """Read a string array (vlen-utf8) into numpy."""
    data = await arr.getitem(slice(None))
    return np.asarray(data)


async def decode_categorical(group: zarr.AsyncGroup) -> pd.Categorical:
    """Decode an AnnData categorical group.

    Structure: group contains 'categories' (string array) and 'codes' (int array).
    """
    categories_arr = await group.getitem("categories")
    codes_arr = await group.getitem("codes")
    categories = await _read_string_array(categories_arr)
    codes = await _read_array(codes_arr)
    return pd.Categorical.from_codes(
        codes=codes,
        categories=[str(c) for c in categories],
    )


async def decode_nullable_string(group: zarr.AsyncGroup) -> np.ndarray:
    """Decode an AnnData nullable-string-array group.

    Structure: group contains 'values' (string array) and 'mask' (bool, True = missing).
    """
    values = await _read_string_array(await group.getitem("values"))
    mask = await _read_array(await group.getitem("mask"))
    out = values.astype(object)
    out[mask.astype(bool)] = None
    return out


async def decode_column(node) -> np.ndarray | pd.Categorical:
    """Decode an obs/var column — could be an array or a categorical group."""
    if isinstance(node, zarr.AsyncGroup):
        attrs = dict(node.attrs)
        if attrs.get("encoding-type") == "categorical":
            return await decode_categorical(node)
        if attrs.get("encoding-type") == "nullable-string-array":
            return await decode_nullable_string(node)
        raise ValueError(f"Unknown column encoding: {attrs.get('encoding-type')}")
    arr = await _read_array(node)
    return arr


async def decode_dataframe(group: zarr.AsyncGroup) -> pd.DataFrame:
    """Decode an AnnData dataframe group.

    Structure: group has _index and column-order in attrs; each column
    is an array or categorical group.
    """
    attrs = dict(group.attrs)
    index_name = attrs.get("_index", "index")
    column_order = attrs.get("column-order", [])

    index = [str(v) for v in np.asarray(await decode_column(await group.getitem(index_name)))]

    columns = {}
    for col_name in column_order:
        try:
            col_node = await group.getitem(col_name)
        except KeyError:
            continue
        col_data = await decode_column(col_node)
        # Wrap object arrays in Series with explicit dtype to preserve None values
        if isinstance(col_data, np.ndarray) and col_data.dtype == object:
            columns[col_name] = pd.Series(col_data, dtype=object)
        else:
            columns[col_name] = col_data

    # Create DataFrame without custom index first to avoid Series realignment
    df = pd.DataFrame(columns)
    df.index = index
    return df


async def decode_sparse_matrix(group: zarr.AsyncGroup) -> scipy.sparse.spmatrix:
    """Decode an AnnData sparse matrix group (csr_matrix or csc_matrix)."""
    attrs = dict(group.attrs)
    encoding = attrs.get("encoding-type")
    shape = attrs.get("shape")
    if not shape:
        raise ValueError("Sparse matrix missing 'shape' attribute")

    data_arr = await group.getitem("data")
    indices_arr = await group.getitem("indices")
    indptr_arr = await group.getitem("indptr")

    data = await _read_array(data_arr)
    indices = await _read_array(indices_arr)
    indptr = await _read_array(indptr_arr)

    if encoding == "csr_matrix":
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))
    elif encoding == "csc_matrix":
        return scipy.sparse.csc_matrix((data, indices, indptr), shape=tuple(shape))
    else:
        raise ValueError(f"Unknown sparse encoding: {encoding}")
