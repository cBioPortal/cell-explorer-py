"""Tests for the --keep-raw option (layers written alongside X)."""
import numpy as np

from cell2zarr._testing import open_zarr, _write_test_h5ad
from cell2zarr.convert import convert_h5ad_to_zarr_chunked
from cell2zarr.models import ConversionConfig


def test_keep_raw_writes_named_layers_only(tmp_path):
    # anndata >= 0.13 reports X as layers[None]; only named layers belong under layers/
    h5ad = tmp_path / "in.h5ad"
    out = tmp_path / "out.zarr"
    path, adata = _write_test_h5ad(h5ad, n_obs=60, n_vars=20)
    adata.layers["counts"] = np.random.randn(60, 20).astype(np.float32)
    adata.write_h5ad(path)

    cfg = ConversionConfig(
        input_file=h5ad, output_file=out, keep_raw=True,
        var_chunk_size=10, cell_chunk_size=25, temp_dir=tmp_path,
    )
    convert_h5ad_to_zarr_chunked(cfg)

    layers = open_zarr(out)["layers"]
    assert sorted(layers.keys()) == ["counts"]
    np.testing.assert_allclose(layers["counts"][:], adata.layers["counts"])
