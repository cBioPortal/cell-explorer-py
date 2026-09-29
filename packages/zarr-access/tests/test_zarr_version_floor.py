"""Regressions pinning the `zarr>=3.4` floor.

Both bugs below are zarr-python's, not ours, and both are invisible from our
side: a store opens or it does not, a facet has values or it silently does not.
Nothing in our code fails, so without these tests a resolver that picks an older
zarr costs us two datasets and the `tissue`/`cell_type` facets again (#186, #187).

Each fails on zarr 3.1.6 and passes on 3.4.0, so they are falsifiable by
construction rather than by assertion.
"""

import base64
import json
import warnings

import pytest
import zarr
from zarr.core.buffer.cpu import Buffer
from zarr.storage import MemoryStore


def _struct_array_metadata() -> dict:
    """A v3 array whose dtype is the `struct` compound other writers emit."""
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        # A struct fill value is the base64 of the packed record, not a scalar.
        "fill_value": base64.b64encode(b"\x00" * 8).decode(),
        "codecs": [{"name": "bytes"}],
        "attributes": {},
        "data_type": {
            "name": "struct",
            "configuration": {
                "fields": [
                    {"name": "B_20", "data_type": "float32"},
                    {"name": "T_cell", "data_type": "float32"},
                ]
            },
        },
    }


@pytest.mark.asyncio
async def test_struct_dtype_array_opens() -> None:
    """#186 — zarr < 3.4 raises `No Zarr data type found that matches ...struct`.

    It happens while materialising node metadata, so a single struct-dtype array
    anywhere in a store took down the whole harvest for that dataset, not just
    that one array.
    """
    store = MemoryStore()
    await store.set(
        "zarr.json",
        Buffer.from_bytes(
            json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}).encode()
        ),
    )
    await store.set(
        "obsm/fractions/zarr.json",
        Buffer.from_bytes(json.dumps(_struct_array_metadata()).encode()),
    )

    array = await zarr.api.asynchronous.open_array(
        store=store, path="obsm/fractions", mode="r"
    )

    assert array.shape == (4,)
    assert array.dtype.fields is not None
    assert list(array.dtype.fields) == ["B_20", "T_cell"]


def test_case_colliding_columns_keep_their_categories() -> None:
    """#187 — zarr < 3.4 drops `categories` from BOTH columns of a case pair.

    A MemoryStore is required: this cannot be reproduced on a case-insensitive
    filesystem, where `obs/Tissue` and `obs/tissue` collide before zarr sees them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = zarr.open_group(MemoryStore(), mode="w", zarr_format=3)
        obs = root.create_group("obs")
        for name in ("Tissue", "tissue", "Stage_Code"):
            column = obs.create_group(name)
            column.create_array("categories", shape=(3,), dtype="int32")
            column.create_array("codes", shape=(5,), dtype="int8")
        zarr.consolidate_metadata(root.store)

        opened = zarr.open_group(root.store, mode="r")
        keys = {n: set(opened["obs"][n].array_keys()) for n in ("Tissue", "tissue", "Stage_Code")}

    # The control: a column with no case-twin was never affected.
    assert keys["Stage_Code"] == {"categories", "codes"}
    # The regression: both halves of the pair keep their categories.
    assert keys["Tissue"] == {"categories", "codes"}
    assert keys["tissue"] == {"categories", "codes"}
