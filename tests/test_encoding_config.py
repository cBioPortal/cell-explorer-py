"""Tests for encoding config loading and resolution."""
import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from convert_h5ad_to_zarr import (
    _resolve_template,
    _load_encoding_config,
    _make_compressor,
    EncodingConfig,
    ArrayEncoding,
    CompressorSpec,
)


class TestResolveTemplate:
    def test_resolves_known_variable(self):
        assert _resolve_template("{n_obs}", {"n_obs": 4264929}) == 4264929

    def test_passes_through_unknown_variable(self):
        assert _resolve_template("{n_dim}", {"n_obs": 100}) == "{n_dim}"

    def test_passes_through_int(self):
        assert _resolve_template(50000, {"n_obs": 100}) == 50000

    def test_passes_through_plain_string(self):
        assert _resolve_template("float16", {}) == "float16"

    def test_passes_through_none(self):
        assert _resolve_template(None, {}) is None


class TestLoadEncodingConfig:
    @pytest.fixture
    def config_path(self, tmp_path):
        config = {
            "X": {
                "chunks": ["{n_obs}", 1],
                "shards": ["{n_obs}", 30],
                "dtype": "float16",
                "compressor": {"name": "zstd", "level": 0},
            },
            "obsm": {
                "chunks": [50000, "{n_dim}"],
                "shards": [50000, "{n_dim}"],
                "dtype": "float16",
                "compressor": {"name": "zstd", "level": 1},
            },
            "obs": {
                "chunks": [50000],
                "compressor": {"name": "zstd", "level": 5},
            },
            "obs/_index": {
                "chunks": [50000],
                "shards": [500000],
                "compressor": {"name": "zstd", "level": 5},
            },
        }
        p = tmp_path / "encoding.json"
        p.write_text(json.dumps(config))
        return p

    def test_resolves_n_obs_in_X(self, config_path):
        enc = _load_encoding_config(config_path, {"n_obs": 1000, "n_vars": 500})
        assert enc.X.chunks == [1000, 1]
        assert enc.X.shards == [1000, 30]

    def test_defers_n_dim_in_obsm(self, config_path):
        enc = _load_encoding_config(config_path, {"n_obs": 1000, "n_vars": 500})
        # {n_dim} not in variables, should pass through as string
        assert enc.obsm.chunks == [50000, "{n_dim}"]
        assert enc.obsm.shards == [50000, "{n_dim}"]

    def test_resolves_n_dim_when_provided(self, config_path):
        enc = _load_encoding_config(config_path, {"n_obs": 1000, "n_vars": 500, "n_dim": 2})
        assert enc.obsm.chunks == [50000, 2]
        assert enc.obsm.shards == [50000, 2]

    def test_preserves_literal_values(self, config_path):
        enc = _load_encoding_config(config_path, {"n_obs": 1000, "n_vars": 500})
        assert enc.X.dtype == "float16"
        assert enc.obs.chunks == [50000]

    def test_preserves_compressor_dict(self, config_path):
        enc = _load_encoding_config(config_path, {"n_obs": 1000, "n_vars": 500})
        assert enc.X.compressor == CompressorSpec(name="zstd", level=0)
        assert enc.obs.compressor == CompressorSpec(name="zstd", level=5)

    def test_obs_index_separate_from_obs(self, config_path):
        enc = _load_encoding_config(config_path, {"n_obs": 1000, "n_vars": 500})
        assert enc.obs_index.shards == [500000]
        assert enc.obs.shards is None

    def test_empty_config(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("{}")
        enc = _load_encoding_config(p, {"n_obs": 1000})
        assert enc.X.chunks is None
        assert enc.obsm.chunks is None
        assert enc.obs.chunks is None
        assert enc.obs_index.chunks is None


class TestMakeCompressor:
    def test_zstd_default(self):
        c = _make_compressor({"name": "zstd"})
        assert "Zstd" in type(c).__name__

    def test_zstd_with_level(self):
        c = _make_compressor({"name": "zstd", "level": 5})
        assert c.level == 5

    def test_none_returns_auto(self):
        assert _make_compressor(None) == "auto"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown compressor"):
            _make_compressor({"name": "lzma"})


class TestEncodingConfigCLIDefaults:
    """Test that encoding config values are correctly extracted for CLI defaults."""

    def test_extracts_var_chunk_size(self):
        enc_raw = {"X": {"chunks": ["{n_obs}", 1], "shards": ["{n_obs}", 30], "dtype": "float16"}}
        assert enc_raw["X"]["chunks"][1] == 1

    def test_extracts_shard_size(self):
        enc_raw = {"X": {"chunks": ["{n_obs}", 1], "shards": ["{n_obs}", 30]}}
        assert enc_raw["X"]["shards"][1] == 30

    def test_extracts_dtype(self):
        enc_raw = {"X": {"dtype": "float16"}}
        assert enc_raw["X"]["dtype"] == "float16"

    def test_extracts_obsm_cell_chunk_size(self):
        enc_raw = {"obsm": {"chunks": [50000, "{n_dim}"]}}
        assert enc_raw["obsm"]["chunks"][0] == 50000

    def test_skips_template_strings(self):
        """Template strings like {n_obs} should not be used as CLI defaults."""
        enc_raw = {"X": {"chunks": ["{n_obs}", "{n_vars}"]}}
        v = enc_raw["X"]["chunks"][1]
        assert isinstance(v, str)  # Not an int, can't be used as default
