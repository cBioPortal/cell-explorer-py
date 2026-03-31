"""CLI entry point for cell2zarr."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from click_default_group import DefaultGroup

from .models import (
    ConversionConfig, EncodingConfig, RunDataset, RunEntry, RunPerformance,
)
from .run_log import (
    read_runs, write_runs, get_next_run_number, update_run_entry,
    compute_output_stats, collect_target_encoding, collect_actual_encoding,
)
from .convert import convert_h5ad_to_zarr, convert_h5ad_to_zarr_chunked


def _setup_logging(run_log: Path | None = None, log_dir: Path | None = None, run_number: int | None = None):
    """Configure cell2zarr logger with stdout and optional file handler."""
    logger = logging.getLogger("cell2zarr")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Always log to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Log to file if run-log is provided
    if run_log and run_number is not None:
        log_dir_path = log_dir or run_log.parent / "logs"
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir_path / f"run-{run_number}.log"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return log_file_path

    return None


def _build_run_hooks(config: ConversionConfig, run_number: int) -> dict:
    """Build hooks dict that wires conversion events to run log updates."""
    log_path = config.run_log

    def on_dataset_read(*, n_obs, n_vars, v_chunk, adata_backed, config, n_cpus):
        obsm_keys = list(adata_backed.obsm.keys()) if adata_backed.obsm else []
        shard_str = [n_obs, config.shard_size] if config.shard_size else None
        update_run_entry(log_path, run_number, {
            "zarr_config": {
                "format": "v3",
                "chunk_shape": [n_obs, v_chunk],
                "sharding": shard_str,
                "dtype": config.dtype,
            },
            "conversion_config": {
                "approach": "two-phase-rechunk",
                "cell_chunk_size": config.cell_chunk_size,
                "temp_var_chunk": min(1000, n_vars),
                "codec_pipeline": "zarrs-python",
                "threads": n_cpus,
                "skip_layers": not config.keep_raw,
                "obsm_keys": obsm_keys,
                "phase2_read_batch": config.shard_size if config.shard_size else v_chunk,
            },
            "dataset": {
                "n_obs": n_obs,
                "n_vars": n_vars,
                "input_size_gb": round(config.input_file.stat().st_size / (1024**3), 1),
                "input_format": "sparse_csr",
            },
        })

    def on_encoding_loaded(*, encoding, config_path):
        encoding_config_raw = json.loads(config_path.read_text())
        update_run_entry(log_path, run_number, {
            "zarr_config": {
                "encoding_config_raw": encoding_config_raw,
                "target_encoding": collect_target_encoding(encoding),
            },
        })

    def on_phase1_done(*, phase1_time):
        update_run_entry(log_path, run_number, {
            "status": "phase2",
            "performance": {"phase1_time_s": round(phase1_time)},
        })

    def on_complete(*, phase1_time, phase2_time, output_path, config, n_obs, n_vars, v_chunk):
        total_time = phase1_time + phase2_time
        output_stats = compute_output_stats(output_path, n_obs, n_vars, config.dtype)
        read_batch = config.shard_size if config.shard_size else v_chunk
        n_batches = (n_vars + read_batch - 1) // read_batch
        phase2_rate = round(n_batches / (phase2_time / 60), 1) if phase2_time > 0 else 0
        actual = collect_actual_encoding(output_path)
        update_run_entry(log_path, run_number, {
            "status": "completed",
            "zarr_config": {"actual_encoding": actual},
            "performance": {
                "end_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "phase1_time_s": round(phase1_time),
                "phase2_time_s": round(phase2_time),
                "total_time_s": round(total_time),
                "phase2_rate_batches_per_min": phase2_rate,
                **output_stats,
            },
        })

    def on_error(*, exception):
        update_run_entry(log_path, run_number, {
            "status": "failed",
            "performance": {"end_time": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "notes": f"Error: {exception}",
        })

    return {
        "on_dataset_read": on_dataset_read,
        "on_encoding_loaded": on_encoding_loaded,
        "on_phase1_done": on_phase1_done,
        "on_complete": on_complete,
        "on_error": on_error,
    }


@click.group(cls=DefaultGroup, default="convert", default_if_no_args=False)
def cli():
    """cell2zarr — h5ad to Zarr conversion toolkit"""


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", required=False, default=None, type=click.Path(path_type=Path))
@click.option("--obs-chunk-size", type=int, help="Observation (cell) chunk size.")
@click.option("--var-chunk-size", type=int, help="Variable (gene) chunk size. Default: 10.")
@click.option("--n-top-genes", type=int, help="Filter to top N highly variable genes.")
@click.option("--keep-raw", is_flag=True, help="Keep raw counts in the output.")
@click.option("--sparse-format", type=click.Choice(["csr", "csc"]), default="csr", help="Sparse matrix format. Default: csr.")
@click.option("--dense", is_flag=True, help="Convert sparse matrix to dense array.")
@click.option("--force-int32", is_flag=True, help="Cast sparse indices to int32.")
@click.option("--two-phase", is_flag=True, help="Two-phase streaming conversion. Low memory usage.")
@click.option("--cell-chunk-size", type=int, default=10000, help="Cells per chunk in phase 1. Default: 10000.")
@click.option("--shard-size", type=int, help="Genes per shard (zarr v3 sharding).")
@click.option("--dtype", type=click.Choice(["float16", "float32", "float64"]), help="Output dtype. Default: float32.")
@click.option("--obsm-cell-chunk-size", type=int, help="Cell chunk size for obsm arrays. Default: 50000.")
@click.option("--run-log", type=click.Path(path_type=Path), help="Path to JSON run log file.")
@click.option("--log-dir", type=click.Path(path_type=Path), help="Directory for stdout log files.")
@click.option("--encoding-config", type=click.Path(exists=True, path_type=Path), help="Path to JSON encoding config.")
@click.option("--temp-dir", type=click.Path(path_type=Path), help="Directory for phase 1 temp zarr.")
def convert(input_file, output_file, obs_chunk_size, var_chunk_size, n_top_genes,
            keep_raw, sparse_format, dense, force_int32, two_phase, cell_chunk_size,
            shard_size, dtype, obsm_cell_chunk_size, run_log, log_dir, encoding_config, temp_dir):
    """Convert h5ad file to Zarr store."""
    if output_file is None:
        output_file = input_file.with_suffix('.zarr')

    if output_file.exists():
        raise click.ClickException(f"Output '{output_file}' already exists")

    if two_phase:
        if sparse_format != "csr" or force_int32:
            click.echo("Note: --sparse-format and --force-int32 are ignored in two-phase mode", err=True)

        enc_defaults = {}
        if encoding_config and encoding_config.exists():
            enc = EncodingConfig.model_validate_json(encoding_config.read_text())
            if enc.X.chunks and len(enc.X.chunks) > 1 and isinstance(enc.X.chunks[1], int):
                enc_defaults["var_chunk_size"] = enc.X.chunks[1]
            if enc.X.shards and len(enc.X.shards) > 1 and isinstance(enc.X.shards[1], int):
                enc_defaults["shard_size"] = enc.X.shards[1]
            if enc.X.dtype:
                enc_defaults["dtype"] = enc.X.dtype
            if enc.obsm.chunks and len(enc.obsm.chunks) > 0 and isinstance(enc.obsm.chunks[0], int):
                enc_defaults["obsm_cell_chunk_size"] = enc.obsm.chunks[0]

        config = ConversionConfig(
            input_file=input_file,
            output_file=output_file,
            var_chunk_size=var_chunk_size if var_chunk_size is not None else enc_defaults.get("var_chunk_size", 10),
            n_top_genes=n_top_genes,
            keep_raw=keep_raw,
            cell_chunk_size=cell_chunk_size,
            shard_size=shard_size if shard_size is not None else enc_defaults.get("shard_size"),
            dtype=dtype if dtype is not None else enc_defaults.get("dtype", "float32"),
            obsm_cell_chunk_size=obsm_cell_chunk_size if obsm_cell_chunk_size is not None else enc_defaults.get("obsm_cell_chunk_size", 50000),
            run_log=run_log,
            log_dir=log_dir,
            encoding_config=encoding_config,
            temp_dir=temp_dir,
        )

        hooks = None
        if config.run_log:
            runs = read_runs(config.run_log)
            run_number = get_next_run_number(runs)

            log_file_path = _setup_logging(config.run_log, config.log_dir, run_number)
            log_file_rel = f"logs/run-{run_number}.log"

            input_size_gb = round(config.input_file.stat().st_size / (1024**3), 1)

            script_args = {
                "input": str(config.input_file),
                "output": str(config.output_file),
                "--two-phase": True,
                "--var-chunk-size": config.var_chunk_size,
                "--keep-raw": config.keep_raw,
                "--cell-chunk-size": config.cell_chunk_size,
                "--dtype": config.dtype,
                "--obsm-cell-chunk-size": config.obsm_cell_chunk_size,
            }
            if config.shard_size is not None:
                script_args["--shard-size"] = config.shard_size
            if config.n_top_genes is not None:
                script_args["--n-top-genes"] = config.n_top_genes
            if config.encoding_config is not None:
                script_args["--encoding-config"] = str(config.encoding_config)

            run_entry = RunEntry(
                run=run_number,
                date=datetime.now().strftime("%Y-%m-%d"),
                status="running",
                script_args=script_args,
                dataset=RunDataset(
                    input_size_gb=input_size_gb,
                    input_format="sparse_csr",
                ),
                performance=RunPerformance(
                    start_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
                notes="",
                log_file=log_file_rel,
            )
            runs.append(run_entry)
            write_runs(config.run_log, runs)

            hooks = _build_run_hooks(config, run_number)
        else:
            _setup_logging()

        logger = logging.getLogger("cell2zarr")
        try:
            convert_h5ad_to_zarr_chunked(config, hooks=hooks)
        finally:
            # Remove all handlers added during this run
            logger.handlers.clear()
    else:
        convert_h5ad_to_zarr(input_file, output_file, obs_chunk_size, var_chunk_size, n_top_genes, keep_raw, sparse_format, force_int32, dense)


@cli.command()
@click.argument("h5ad_file", type=click.Path(exists=True, path_type=Path))
@click.argument("zarr_store", type=click.Path(path_type=Path))
@click.option("--key", required=True, help="Key to add (e.g. obsm, obsm/X_umap, obs, X, layers/counts).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing keys.")
@click.option("--encoding-config", type=click.Path(exists=True, path_type=Path), help="Path to JSON encoding config.")
@click.option("--dtype", type=click.Choice(["float16", "float32", "float64"]), default="float32", help="Target dtype. Default: float32.")
@click.option("--temp-dir", type=click.Path(path_type=Path), help="Temp directory for large keys (X, layers).")
def add(h5ad_file, zarr_store, key, overwrite, encoding_config, dtype, temp_dir):
    """Add a key from h5ad to an existing Zarr store."""
    from .convert import add_key_to_store
    from .encoding import load_encoding_config

    encoding = None
    if encoding_config and encoding_config.exists():
        encoding = load_encoding_config(encoding_config, {})

    add_key_to_store(
        h5ad_path=h5ad_file,
        zarr_path=zarr_store,
        key=key,
        overwrite=overwrite,
        dtype=dtype,
        encoding=encoding,
        temp_dir=temp_dir,
    )
