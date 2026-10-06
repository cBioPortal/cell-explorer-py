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


ATLAS_LIKE = {
    "obs": {"chunks": ["{n_obs}"], "shards": ["{n_obs}"], "compressor": {"name": "zstd", "level": 5}},
    "obs/_index": {"chunks": ["{n_obs}"], "shards": ["{n_obs}"], "compressor": {"name": "zstd", "level": 5}},
}


def _h5ad_with_columns(path: Path, n_obs: int = 60) -> ad.AnnData:
    """Numeric, categorical (2 categories), bool and all-unique categorical obs columns."""
    adata = _write_test_h5ad(path, n_obs=n_obs, n_vars=20)[1]
    adata.obs["score"] = np.arange(n_obs, dtype=np.float32)
    adata.obs["flag"] = np.arange(n_obs) % 2 == 0
    adata.obs["barcode"] = pd.Categorical([f"bc{i}" for i in range(n_obs)])
    adata.write_h5ad(path)
    return adata


def _convert(tmp_path: Path, encoding: dict | None, n_obs: int = 60, obsm_cell_chunk_size: int = 50000) -> tuple[Path, ad.AnnData]:
    h5ad = tmp_path / "in.h5ad"
    adata = _h5ad_with_columns(h5ad, n_obs)
    cfg_path = None
    if encoding is not None:
        cfg_path = tmp_path / "encoding.json"
        cfg_path.write_text(json.dumps(encoding))
    out = tmp_path / "out.zarr"
    convert_h5ad_to_zarr_chunked(ConversionConfig(
        input_file=h5ad, output_file=out, var_chunk_size=10, cell_chunk_size=25,
        temp_dir=tmp_path, encoding_config=cfg_path, obsm_cell_chunk_size=obsm_cell_chunk_size,
    ))
    return out, adata


def test_obs_config_applies_to_n_obs_length_arrays(tmp_path):
    out, _ = _convert(tmp_path, ATLAS_LIKE)
    for path in ["obs/score", "obs/flag", "obs/cell_type/codes", "obs/barcode/codes", "obs/barcode/categories"]:
        assert _layout(out, path) == {"shards": [60], "chunks": [60], "zstd_level": 5}, path


def test_obs_config_does_not_pad_small_categories(tmp_path):
    out, _ = _convert(tmp_path, ATLAS_LIKE)
    layout = _layout(out, "obs/cell_type/categories")
    assert layout["shards"] is None
    assert layout["chunks"] == [2]


def test_obs_index_config_applies_to_values_and_mask(tmp_path):
    out, _ = _convert(tmp_path, ATLAS_LIKE)
    for part in ["values", "mask"]:
        assert _layout(out, f"obs/_index/{part}") == {"shards": [60], "chunks": [60], "zstd_level": 5}, part


def test_without_config_index_aligns_to_obsm_chunks_and_columns_untouched(tmp_path):
    out, _ = _convert(tmp_path, None, obsm_cell_chunk_size=25)
    for part in ["values", "mask"]:
        assert _layout(out, f"obs/_index/{part}")["chunks"] == [25], part
    assert _layout(out, "obs/score")["shards"] is None


def test_compressor_only_config_keeps_column_chunking(tmp_path):
    out, _ = _convert(tmp_path, {"obs": {"compressor": {"name": "zstd", "level": 7}}})
    (tmp_path / "ref").mkdir()
    reference, _ = _convert(tmp_path / "ref", None)
    layout = _layout(out, "obs/score")
    assert layout["zstd_level"] == 7
    assert layout["chunks"] == _layout(reference, "obs/score")["chunks"]


def test_reencoded_obs_round_trips(tmp_path):
    out, adata = _convert(tmp_path, ATLAS_LIKE)
    obs = ad.read_zarr(out).obs
    assert obs.index.tolist() == adata.obs.index.tolist()
    np.testing.assert_array_equal(obs["score"].to_numpy(), adata.obs["score"].to_numpy())
    assert obs["flag"].tolist() == adata.obs["flag"].tolist()
    assert obs["barcode"].tolist() == adata.obs["barcode"].tolist()
    assert obs["cell_type"].tolist() == adata.obs["cell_type"].astype(str).tolist()


def _load(cfg_path: Path, n_obs: int):
    from cell2zarr.encoding import load_encoding_config
    return load_encoding_config(cfg_path, {"n_obs": n_obs, "n_vars": 20})


def test_add_obs_matches_full_convert_layout(tmp_path):
    from cell2zarr.convert import add_key_to_store

    out, _ = _convert(tmp_path, ATLAS_LIKE)
    cfg_path = tmp_path / "encoding.json"
    add_key_to_store(tmp_path / "in.h5ad", out, key="obs", overwrite=True, encoding=_load(cfg_path, 60))
    for path in ["obs/score", "obs/cell_type/codes", "obs/_index/values", "obs/_index/mask"]:
        assert _layout(out, path) == {"shards": [60], "chunks": [60], "zstd_level": 5}, path


def _add_obs_keeps_layout(tmp_path: Path, encoding: dict | None) -> None:
    from cell2zarr.convert import add_key_to_store

    out, _ = _convert(tmp_path, encoding)
    paths = ["obs/_index/values", "obs/_index/mask", "obs/score", "obs/cell_type/codes"]
    before = {p: _layout(out, p) for p in paths}
    cfg = _load(tmp_path / "encoding.json", 60) if encoding is not None else None
    add_key_to_store(tmp_path / "in.h5ad", out, key="obs", overwrite=True, encoding=cfg)
    assert {p: _layout(out, p) for p in paths} == before


def test_add_obs_without_config_matches_full_convert_layout(tmp_path):
    _add_obs_keeps_layout(tmp_path, None)


def test_add_obs_with_obs_only_config_matches_full_convert_layout(tmp_path):
    _add_obs_keeps_layout(tmp_path, {"obs": {"compressor": {"name": "zstd", "level": 7}}})


def test_add_obs_uses_config_obsm_chunk_for_index(tmp_path):
    from cell2zarr.convert import add_key_to_store

    encoding = {"obsm": {"chunks": [25, "{n_dim}"]}}
    # The CLI feeds obsm.chunks[0] to a full convert as obsm_cell_chunk_size.
    out, _ = _convert(tmp_path, encoding, obsm_cell_chunk_size=25)
    cfg = _load(tmp_path / "encoding.json", 60)
    full = {p: _layout(out, p) for p in ["obs/_index/values", "obs/_index/mask"]}
    assert full["obs/_index/values"]["chunks"] == [25]
    add_key_to_store(tmp_path / "in.h5ad", out, key="obs", overwrite=True, encoding=cfg)
    assert {p: _layout(out, p) for p in full} == full
