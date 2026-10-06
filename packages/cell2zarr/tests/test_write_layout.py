"""On-disk layout of cell2zarr output: cell2zarr, not anndata, decides chunking and sharding."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cell2zarr._testing import _write_test_h5ad
from cell2zarr.convert import convert_h5ad_to_zarr, convert_h5ad_to_zarr_chunked
from cell2zarr.models import ConversionConfig


def _layout(store: Path, path: str) -> dict:
    """Chunks, shards and zstd level of one array, read from its zarr.json."""
    meta = json.loads((Path(store) / path / "zarr.json").read_text())
    codecs = meta["codecs"]
    grid = meta["chunk_grid"]["configuration"]["chunk_shape"]
    if codecs[0]["name"] == "sharding_indexed":
        inner = codecs[0]["configuration"]
        shards, chunks, codecs = grid, inner["chunk_shape"], inner["codecs"]
    else:
        shards, chunks = None, grid
    zstd = next((c for c in codecs if c["name"] == "zstd"), None)
    return {
        "shards": shards,
        "chunks": chunks,
        "zstd_level": zstd["configuration"]["level"] if zstd else None,
    }


def _arrays(store: Path, under: str) -> list[str]:
    """Every array path below `under`, relative to the store root."""
    root = Path(store)
    return sorted(
        str(p.parent.relative_to(root))
        for p in (root / under).rglob("zarr.json")
        if json.loads(p.read_text())["node_type"] == "array"
    )


def test_two_phase_does_not_shard_anndata_written_arrays(tmp_path):
    h5ad = tmp_path / "in.h5ad"
    _write_test_h5ad(h5ad, n_obs=60, n_vars=20)
    out = tmp_path / "out.zarr"
    convert_h5ad_to_zarr_chunked(ConversionConfig(
        input_file=h5ad, output_file=out, var_chunk_size=10, cell_chunk_size=25, temp_dir=tmp_path,
    ))
    for path in _arrays(out, "obs") + _arrays(out, "var"):
        assert _layout(out, path)["shards"] is None, path


def test_in_memory_convert_does_not_shard(tmp_path):
    h5ad = tmp_path / "in.h5ad"
    _write_test_h5ad(h5ad, n_obs=60, n_vars=20)
    out = tmp_path / "out.zarr"
    convert_h5ad_to_zarr(h5ad, out)
    for path in _arrays(out, "obs") + _arrays(out, "var") + _arrays(out, "X"):
        assert _layout(out, path)["shards"] is None, path


def test_conversion_restores_anndata_settings(tmp_path):
    h5ad = tmp_path / "in.h5ad"
    _write_test_h5ad(h5ad, n_obs=60, n_vars=20)
    # Pin a known starting state: other tests in the session may set these globally.
    with ad.settings.override(allow_write_nullable_strings=False, auto_shard_zarr_v3=True):
        convert_h5ad_to_zarr_chunked(ConversionConfig(
            input_file=h5ad, output_file=tmp_path / "out.zarr", var_chunk_size=10, cell_chunk_size=25, temp_dir=tmp_path,
        ))
        convert_h5ad_to_zarr(h5ad, tmp_path / "mem.zarr")
        assert ad.settings.allow_write_nullable_strings is False
        assert ad.settings.auto_shard_zarr_v3 is True


def test_index_is_written_as_nullable_string(tmp_path):
    h5ad = tmp_path / "in.h5ad"
    _write_test_h5ad(h5ad, n_obs=60, n_vars=20)
    out = tmp_path / "out.zarr"
    convert_h5ad_to_zarr_chunked(ConversionConfig(
        input_file=h5ad, output_file=out, var_chunk_size=10, cell_chunk_size=25, temp_dir=tmp_path,
    ))
    index = json.loads((out / "obs" / "_index" / "zarr.json").read_text())
    assert index["node_type"] == "group"
    assert index["attributes"]["encoding-type"] == "nullable-string-array"
