"""Run log persistence and stats collection."""
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .models import EncodingConfig, RunEntry


class LogTee:
    """Tee stdout to both terminal and a log file."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def read_runs(log_path: Path) -> list[RunEntry]:
    """Read JSON array from run log file."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return []
    with open(log_path) as f:
        return [RunEntry.model_validate(r) for r in json.load(f)]


def write_runs(log_path: Path, runs: list[RunEntry]) -> None:
    """Write JSON array to run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump([r.model_dump(exclude_none=True) for r in runs], f, indent=2)
        f.write("\n")


def get_next_run_number(runs: list[RunEntry]) -> int:
    """Get the next run number."""
    if not runs:
        return 1
    return max(r.run for r in runs) + 1


def update_run_entry(log_path: Path, run_number: int, updates: dict) -> None:
    """Update fields on an existing run entry and write back to disk."""
    runs = read_runs(log_path)
    for i, r in enumerate(runs):
        if r.run == run_number:
            d = r.model_dump(exclude_none=True)
            for key, val in updates.items():
                if isinstance(val, dict) and isinstance(d.get(key), dict):
                    d[key].update(val)
                else:
                    d[key] = val
            runs[i] = RunEntry.model_validate(d)
            break
    write_runs(log_path, runs)


def compute_output_stats(output_path: Path, n_obs: int, n_vars: int, dtype: str) -> dict:
    """Compute output zarr stats: size, shard count, compression ratio."""
    # Total output size
    total_bytes = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    )
    output_size_gb = round(total_bytes / (1024**3), 1)

    # X subtree stats
    x_path = output_path / "X"
    x_files = [f for f in x_path.rglob("*") if f.is_file() and f.name != "zarr.json"]
    x_bytes = sum(f.stat().st_size for f in x_files)
    n_shard_files = len(x_files)
    avg_shard_size_mb = round(x_bytes / n_shard_files / (1024**2), 1) if n_shard_files else 0

    # Compression ratio: uncompressed size / on-disk X size
    element_size = np.dtype(dtype).itemsize
    uncompressed_bytes = n_obs * n_vars * element_size
    compression_ratio = round(uncompressed_bytes / x_bytes, 1) if x_bytes else 0

    return {
        "output_size_gb": output_size_gb,
        "n_shard_files": n_shard_files,
        "avg_shard_size_mb": avg_shard_size_mb,
        "compression_ratio": compression_ratio,
    }


def collect_target_encoding(encoding: EncodingConfig) -> dict[str, Any]:
    """Serialize the resolved encoding config for the run log."""
    result = {}
    for key, section in [("X", encoding.X), ("obsm", encoding.obsm),
                         ("obs", encoding.obs), ("obs/_index", encoding.obs_index)]:
        d = section.model_dump(exclude_none=True)
        if d:
            # Serialize CompressorSpec to plain dict
            if "compressor" in d and isinstance(d["compressor"], dict):
                d["compressor"] = {k: v for k, v in d["compressor"].items()
                                   if k == "name" or (k == "level" and v != 0)}
            result[key] = d
    return result


def collect_actual_encoding(output_path: Path) -> dict[str, Any]:
    """Read back actual zarr metadata for each array in the output store."""
    store = zarr.open(str(output_path), mode="r")
    result = {}

    def _dir_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    def _array_info(arr, arr_path: Path) -> dict[str, Any]:
        info: dict[str, Any] = {
            "shape": list(arr.shape),
            "chunks": list(arr.chunks),
            "dtype": str(arr.dtype),
        }
        if arr.shards is not None:
            info["shards"] = list(arr.shards)
        size_bytes = _dir_size(arr_path)
        if size_bytes >= 1e9:
            info["size_gb"] = round(size_bytes / 1e9, 2)
        else:
            info["size_mb"] = round(size_bytes / 1e6, 1)
        return info

    # X
    if "X" in store:
        result["X"] = _array_info(store["X"], output_path / "X")

    # obsm arrays
    if "obsm" in store:
        for key in sorted(store["obsm"].keys()):
            try:
                arr = store[f"obsm/{key}"]
                if hasattr(arr, "shape"):
                    result[f"obsm/{key}"] = _array_info(arr, output_path / "obsm" / key)
            except Exception:
                pass

    # obs/_index
    if "obs" in store and "_index" in store["obs"]:
        result["obs/_index"] = _array_info(store["obs/_index"], output_path / "obs" / "_index")

    return result
