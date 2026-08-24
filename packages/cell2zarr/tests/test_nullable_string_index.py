"""Two-phase conversion of an h5ad whose obs index is a nullable string array.

anndata stores a pandas nullable-string index under the `nullable-string-array`
encoding: a zarr *group* holding `values` and `mask`, rather than a flat array.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell2zarr.convert import convert_h5ad_to_zarr_chunked
from cell2zarr.models import ConversionConfig
from cell2zarr._testing import open_zarr

N_OBS = 8
N_VARS = 4


@pytest.fixture
def nullable_index_h5ad(tmp_path):
    """An h5ad whose obs/_index is written as a nullable-string-array group."""
    ad.settings.allow_write_nullable_strings = True

    adata = ad.AnnData(
        X=np.arange(N_OBS * N_VARS, dtype=np.float32).reshape(N_OBS, N_VARS),
        obs=pd.DataFrame(
            {"score": np.arange(N_OBS, dtype=np.float32)},
            index=pd.Index([f"cell_{i}" for i in range(N_OBS)], dtype="string"),
        ),
        var=pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(N_VARS)], dtype=object)),
    )
    adata.obsm["X_umap"] = np.zeros((N_OBS, 2), dtype=np.float32)

    path = tmp_path / "nullable.h5ad"
    adata.write_h5ad(path)
    return path


def test_two_phase_convert_accepts_a_nullable_string_index(nullable_index_h5ad, tmp_path):
    out = tmp_path / "out.zarr"
    convert_h5ad_to_zarr_chunked(
        ConversionConfig(
            input_file=nullable_index_h5ad,
            output_file=out,
            var_chunk_size=1,
            cell_chunk_size=4,
            temp_dir=tmp_path,
        )
    )

    assert out.exists()


def test_nullable_string_index_values_survive_conversion(nullable_index_h5ad, tmp_path):
    out = tmp_path / "out.zarr"
    convert_h5ad_to_zarr_chunked(
        ConversionConfig(
            input_file=nullable_index_h5ad,
            output_file=out,
            var_chunk_size=1,
            cell_chunk_size=4,
            temp_dir=tmp_path,
        )
    )

    names = ad.read_zarr(out).obs.index.tolist()
    assert names == [f"cell_{i}" for i in range(N_OBS)]
