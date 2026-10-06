"""Generate nullable_strings.zarr: an AnnData store using anndata 0.13's nullable-string encoding.

Run from the repo root: uv run --project packages/zarr-access python packages/zarr-access/tests/generate_nullable_string_fixture.py
"""
from pathlib import Path
import shutil

import anndata as ad
import numpy as np
import numpy.dtypes
import pandas as pd
import zarr

dst = Path(__file__).parent / "fixtures" / "nullable_strings.zarr"
if dst.exists():
    shutil.rmtree(dst)

obs = pd.DataFrame(
    {
        "donor": pd.array(["d1", None, "d2", "d1", None, "d2"], dtype="string"),
        "cell_type": pd.Categorical(["T", "B", "T", "B", "T", "B"]),
        "score": np.arange(6, dtype=np.float32),
    },
    index=pd.Index([f"cell_{i}" for i in range(6)], dtype="string"),
)
var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(3)], dtype="string"))
adata = ad.AnnData(X=np.arange(18, dtype=np.float32).reshape(6, 3), obs=obs, var=var)

with ad.settings.override(allow_write_nullable_strings=True, auto_shard_zarr_v3=False):
    adata.write_zarr(dst)
zarr.consolidate_metadata(str(dst))

# Convert donor column from categorical to nullable-string-array
store = zarr.open_group(str(dst), mode="r+")
donor_categorical = store["obs/donor"]
codes = donor_categorical["codes"][:]
categories = donor_categorical["categories"][:]

# Create values array: map codes to categories, with -1 → None (via mask)
values = np.array([categories[c] if c >= 0 else "" for c in codes], dtype=object)
mask = codes < 0

# Delete the categorical group and create nullable-string-array
shutil.rmtree(str(dst / "obs/donor"))

# Create the new nullable-string-array structure
donor_group = store.create_group("obs/donor")
donor_group.attrs["encoding-type"] = "nullable-string-array"
# Use from_array to properly infer the StringDType
values_arr = np.array(values, dtype=numpy.dtypes.StringDType())
donor_group.create_array("values", data=values_arr)
donor_group.create_array("mask", data=mask)

zarr.consolidate_metadata(str(dst))

for path in ["obs/_index", "var/_index", "obs/donor"]:
    enc = zarr.open_group(str(dst / path), mode="r").attrs["encoding-type"]
    assert enc == "nullable-string-array", (path, enc)
print(f"Generated {dst}")
