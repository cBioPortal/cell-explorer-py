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


async def decode_column(node) -> np.ndarray | pd.Categorical:
    """Decode an obs/var column — could be an array or a categorical group."""
    if isinstance(node, zarr.AsyncGroup):
        attrs = dict(node.attrs)
        if attrs.get("encoding-type") == "categorical":
            return await decode_categorical(node)
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

    index_node = await group.getitem(index_name)
    index = await _read_string_array(index_node)
    index = [str(v) for v in index]

    columns = {}
    for col_name in column_order:
        try:
            col_node = await group.getitem(col_name)
        except KeyError:
            continue
        columns[col_name] = await decode_column(col_node)

    return pd.DataFrame(columns, index=index)


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
