"""Pydantic models for cell2zarr."""
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CompressorSpec(BaseModel):
    """Compressor specification for zarr arrays."""
    name: str
    level: int = 0
    cname: str = "lz4"  # blosc only


class ArrayEncoding(BaseModel):
    """Encoding specification for a zarr array."""
    chunks: list[int | str] | None = None
    shards: list[int | str] | None = None
    dtype: str | None = None
    compressor: CompressorSpec | None = None


class EncodingConfig(BaseModel):
    """Full encoding config for all array types."""
    X: ArrayEncoding = Field(default_factory=ArrayEncoding)
    obsm: ArrayEncoding = Field(default_factory=ArrayEncoding)
    obs: ArrayEncoding = Field(default_factory=ArrayEncoding)
    obs_index: ArrayEncoding = Field(default_factory=ArrayEncoding, alias="obs/_index")

    model_config = {"populate_by_name": True}


class ConversionConfig(BaseModel):
    """Configuration for chunked h5ad-to-zarr conversion."""
    input_file: Path
    output_file: Path
    var_chunk_size: int = 10
    n_top_genes: int | None = None
    keep_raw: bool = False
    normalize: bool = False
    cell_chunk_size: int = 10000
    shard_size: int | None = None
    dtype: str = "float32"
    obsm_cell_chunk_size: int = 50000
    run_db: Path | None = None
    encoding_config: Path | None = None
    temp_dir: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"float16", "float32", "float64"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got '{v}'")
        return v


RunStatus = Literal["running", "phase2", "completed", "failed", "cancelled", "killed"]


class RunDataset(BaseModel):
    """Dataset info for a conversion run."""
    model_config = {"extra": "allow"}
    n_obs: int | None = None
    n_vars: int | None = None
    input_size_gb: float | None = None
    input_format: str | None = None


class RunPerformance(BaseModel):
    """Performance metrics for a conversion run."""
    model_config = {"extra": "allow"}
    start_time: str | None = None
    end_time: str | None = None
    phase1_time_s: int | None = None
    phase2_time_s: int | None = None
    total_time_s: int | None = None
    output_size_gb: float | None = None
    avg_chunk_size_mb: float | None = None
    compression_ratio: float | None = None
    phase2_rate_batches_per_min: float | None = None


class RunZarrConfig(BaseModel):
    """Zarr output config for a conversion run."""
    model_config = {"extra": "allow"}
    format: str | None = None
    chunk_shape: list[int] | None = None
    sharding: list[int] | None = None
    dtype: str | None = None
    compression: str | None = None
    target_encoding: dict[str, Any] | None = None
    actual_encoding: dict[str, Any] | None = None


class RunConversionConfig(BaseModel):
    """Conversion config snapshot for a run."""
    model_config = {"extra": "allow"}
    approach: str | None = None
    cell_chunk_size: int | None = None
    temp_var_chunk: int | None = None
    codec_pipeline: str | None = None
    threads: int | None = None
    skip_layers: bool | None = None
    obsm_keys: list[str] | None = None
    phase2_read_batch: int | None = None


class RunEntry(BaseModel):
    """A single entry in the conversion run log."""
    model_config = {"extra": "allow"}
    run: int
    date: str | None = None
    status: RunStatus = "running"
    script_args: dict[str, Any] | None = None
    zarr_config: RunZarrConfig | None = None
    conversion_config: RunConversionConfig | None = None
    dataset: RunDataset | None = None
    performance: RunPerformance | None = None
    notes: str = ""
    log_file: str | None = None
