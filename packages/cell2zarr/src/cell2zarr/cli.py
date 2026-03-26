"""CLI entry point for cell2zarr."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    ConversionConfig, EncodingConfig, RunDataset, RunEntry, RunPerformance,
)
from .run_log import (
    LogTee, read_runs, write_runs, get_next_run_number, update_run_entry,
    compute_output_stats, collect_target_encoding, collect_actual_encoding,
)
from .convert import convert_h5ad_to_zarr, convert_h5ad_to_zarr_chunked


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


def _handle_convert(args):
    """Handle the convert subcommand (or legacy positional invocation)."""
    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Set output file if not provided
    if args.output_file is None:
        args.output_file = args.input_file.with_suffix('.zarr')

    # Check if output already exists
    if args.output_file.exists():
        print(f"Error: Output '{args.output_file}' already exists", file=sys.stderr)
        sys.exit(1)

    # Convert
    if args.two_phase:
        if args.sparse_format != "csr" or args.force_int32:
            print("Note: --sparse-format and --force-int32 are ignored in two-phase mode", file=sys.stderr)

        # Apply encoding config defaults for CLI args not explicitly set
        enc_defaults = {}
        if args.encoding_config and args.encoding_config.exists():
            enc = EncodingConfig.model_validate_json(args.encoding_config.read_text())
            if enc.X.chunks and len(enc.X.chunks) > 1 and isinstance(enc.X.chunks[1], int):
                enc_defaults["var_chunk_size"] = enc.X.chunks[1]
            if enc.X.shards and len(enc.X.shards) > 1 and isinstance(enc.X.shards[1], int):
                enc_defaults["shard_size"] = enc.X.shards[1]
            if enc.X.dtype:
                enc_defaults["dtype"] = enc.X.dtype
            if enc.obsm.chunks and len(enc.obsm.chunks) > 0 and isinstance(enc.obsm.chunks[0], int):
                enc_defaults["obsm_cell_chunk_size"] = enc.obsm.chunks[0]

        config = ConversionConfig(
            input_file=args.input_file,
            output_file=args.output_file,
            var_chunk_size=args.var_chunk_size if args.var_chunk_size is not None else enc_defaults.get("var_chunk_size", 10),
            n_top_genes=args.n_top_genes,
            keep_raw=args.keep_raw,
            cell_chunk_size=args.cell_chunk_size,
            shard_size=args.shard_size if args.shard_size is not None else enc_defaults.get("shard_size"),
            dtype=args.dtype if args.dtype is not None else enc_defaults.get("dtype", "float32"),
            obsm_cell_chunk_size=args.obsm_cell_chunk_size if args.obsm_cell_chunk_size is not None else enc_defaults.get("obsm_cell_chunk_size", 50000),
            run_log=args.run_log,
            log_dir=args.log_dir,
            encoding_config=args.encoding_config,
            temp_dir=args.temp_dir,
        )

        # Set up run logging and hooks if --run-log provided
        hooks = None
        log_tee = None
        if config.run_log:
            runs = read_runs(config.run_log)
            run_number = get_next_run_number(runs)

            log_dir = config.log_dir or config.run_log.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / f"run-{run_number}.log"
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

            log_tee = LogTee(log_file_path)
            sys.stdout = log_tee

            hooks = _build_run_hooks(config, run_number)

        try:
            convert_h5ad_to_zarr_chunked(config, hooks=hooks)
        finally:
            if log_tee is not None:
                log_tee.close()
    else:
        convert_h5ad_to_zarr(args.input_file, args.output_file, args.obs_chunk_size, args.var_chunk_size, args.n_top_genes, args.keep_raw, args.sparse_format, args.force_int32, args.dense)


def _handle_add(args):
    """Handle the add subcommand."""
    from .convert import add_key_to_store
    from .encoding import load_encoding_config

    encoding = None
    if args.encoding_config and args.encoding_config.exists():
        encoding = load_encoding_config(args.encoding_config, {})

    add_key_to_store(
        h5ad_path=args.h5ad_file,
        zarr_path=args.zarr_store,
        key=args.key,
        overwrite=args.overwrite,
        dtype=args.dtype,
        encoding=encoding,
        temp_dir=args.temp_dir,
    )


def main():
    known_subcommands = {"convert", "add"}
    if len(sys.argv) > 1 and sys.argv[1] not in known_subcommands and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "convert")

    parser = argparse.ArgumentParser(
        description="Convert h5ad file to zarr format"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- convert subparser ---
    convert_parser = subparsers.add_parser("convert", help="Convert h5ad file to Zarr store")
    convert_parser.add_argument(
        "input_file",
        type=Path,
        help="Input h5ad file"
    )
    convert_parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Output zarr directory (defaults to input filename with .zarr extension)"
    )
    convert_parser.add_argument(
        "--obs-chunk-size",
        type=int,
        help="Observation (cell) chunk size. If not specified, uses full dataset size."
    )
    convert_parser.add_argument(
        "--var-chunk-size",
        type=int,
        help="Variable (gene) chunk size. Default: 10. Overrides encoding config."
    )
    convert_parser.add_argument(
        "--n-top-genes",
        type=int,
        help="Filter to top N highly variable genes before writing to zarr"
    )
    convert_parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw counts in the output (default: remove to save space)"
    )
    convert_parser.add_argument(
        "--sparse-format",
        choices=["csr", "csc"],
        default="csr",
        help="Sparse matrix format: csr (row-oriented, good for cell access) or csc (column-oriented, good for gene access). Default: csr"
    )
    convert_parser.add_argument(
        "--dense",
        action="store_true",
        help="Convert sparse matrix to dense array before writing to zarr"
    )
    convert_parser.add_argument(
        "--force-int32",
        action="store_true",
        help="Cast sparse matrix indices/indptr to int32 for JavaScript/Vitessce compatibility. Will fail if values exceed int32 max."
    )
    convert_parser.add_argument(
        "--two-phase",
        action="store_true",
        help="Use two-phase streaming conversion: reads h5ad in backed mode, writes row-chunked temp zarr, then rechunks to final column layout. Low memory usage."
    )
    convert_parser.add_argument(
        "--cell-chunk-size",
        type=int,
        default=10000,
        help="Number of cells per chunk in phase 1 streaming. Default: 10000"
    )
    convert_parser.add_argument(
        "--shard-size",
        type=int,
        help="Number of genes per shard (zarr v3 sharding). If not set, no sharding is used."
    )
    convert_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "float64"],
        help="Data type for the output X matrix and obsm embeddings. Default: float32. Overrides encoding config."
    )
    convert_parser.add_argument(
        "--obsm-cell-chunk-size",
        type=int,
        help="Cell chunk size for obsm embedding arrays (enables progressive loading). Default: 50000. Overrides encoding config."
    )
    convert_parser.add_argument(
        "--run-log",
        type=Path,
        help="Path to JSON run log file (e.g. docs/conversion-runs.json). No logging if omitted."
    )
    convert_parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for stdout log files. Defaults to <run-log-dir>/logs/"
    )
    convert_parser.add_argument(
        "--encoding-config",
        type=Path,
        help="Path to JSON encoding config file specifying chunks, shards, dtype, and compressor for X, obsm, and obs arrays."
    )
    convert_parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Directory for phase 1 temp zarr. Defaults to system /tmp. Use a large disk if /tmp is too small."
    )

    # --- add subparser ---
    add_parser = subparsers.add_parser("add", help="Add a key from h5ad to an existing Zarr store")
    add_parser.add_argument("h5ad_file", type=Path, help="Input h5ad file")
    add_parser.add_argument("zarr_store", type=Path, help="Existing Zarr store directory")
    add_parser.add_argument("--key", required=True, help="Key to add (e.g. obsm, obsm/X_umap, obs, X, layers/counts)")
    add_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing keys")
    add_parser.add_argument("--encoding-config", type=Path, help="Path to JSON encoding config")
    add_parser.add_argument("--dtype", type=str, choices=["float16", "float32", "float64"], default="float32", help="Target dtype. Default: float32")
    add_parser.add_argument("--temp-dir", type=Path, help="Temp directory for large keys (X, layers)")

    args = parser.parse_args()

    if args.command == "add":
        _handle_add(args)
    else:
        _handle_convert(args)


if __name__ == "__main__":
    main()
