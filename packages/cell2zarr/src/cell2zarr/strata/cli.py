"""click subcommand `cell2zarr build-strata`.

Wires CLI flags + optional YAML config → StrataConfig → build_strata().
Custom error rendering for StrataError subclasses so the user never sees a
Python traceback for known errors.
"""
import sys
from pathlib import Path

import click

from cell2zarr.strata.build import build_strata
from cell2zarr.strata.config import StrataConfig
from cell2zarr.strata.exceptions import (
    StrataError,
    StrataConfigError,
)
from cell2zarr.strata.io import open_dataset
from cell2zarr.strata.validate import list_categorical_obs_columns


def _format_categorical_help(dataset_zarr: Path) -> str:
    """Render the helpful 'available obs columns' block for missing-axes errors."""
    try:
        root = open_dataset(dataset_zarr)
        cols = list_categorical_obs_columns(root)
    except Exception:
        return ""
    if not cols:
        return ""
    lines = ["", "Available categorical columns in obs/:"]
    for name, cardinality in cols.items():
        lines.append(f"  {name:<24} ({cardinality} unique values)")
    lines.append("")
    lines.append("Example:")
    lines.append(
        f"  cell2zarr build-strata {dataset_zarr} "
        f"--atomic-axes {' '.join(list(cols)[:3])}"
    )
    return "\n".join(lines)


def _parse_coarse_spec(spec: str) -> list[str]:
    """Each --coarse argument is either 'col' or 'col1,col2,col3'."""
    return [c.strip() for c in spec.split(",") if c.strip()]


@click.command("build-strata")
@click.argument("dataset_zarr", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML config file. CLI flags override YAML field-by-field.",
)
@click.option(
    "--atomic-axes",
    "atomic_axes",
    multiple=True,
    help="obs/ column names that define atomic strata. Repeat or list inline.",
)
@click.option(
    "--coarse",
    "coarse",
    multiple=True,
    help="Coarse table spec: a single column ('cell_type') or comma-separated list ('cell_type,treatment').",
)
@click.option(
    "--max-atomic-cardinality",
    type=int,
    default=None,
    help="Reject the build if estimated |atomic strata| exceeds this. Default 50000.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing uns/strata/* groups.",
)
def build_strata_command(
    dataset_zarr: Path,
    config_path: Path | None,
    atomic_axes: tuple[str, ...],
    coarse: tuple[str, ...],
    max_atomic_cardinality: int | None,
    force: bool,
):
    """Build precomputed strata tables for an AnnData zarr store."""
    # Start from YAML defaults (if any), then layer CLI flags on top.
    if config_path is not None:
        config_dict = StrataConfig.from_yaml(config_path).model_dump()
    else:
        config_dict = {}

    if atomic_axes:
        config_dict["atomic_axes"] = list(atomic_axes)
    if coarse:
        config_dict["coarse"] = [_parse_coarse_spec(spec) for spec in coarse]
    if max_atomic_cardinality is not None:
        config_dict["max_atomic_cardinality"] = max_atomic_cardinality
    if force:
        config_dict["force"] = True

    if not config_dict.get("atomic_axes"):
        msg = "Error: --atomic-axes is required (or set atomic_axes in --config)."
        click.echo(msg)
        click.echo(_format_categorical_help(dataset_zarr))
        sys.exit(2)

    try:
        config = StrataConfig.model_validate(config_dict)
    except Exception as e:  # pydantic ValidationError, etc.
        click.echo(f"Error: invalid config: {e}")
        sys.exit(2)

    try:
        build_strata(dataset_zarr, config)
    except StrataError as e:
        click.echo(f"Error: {e}")
        # Special-case: missing-column config error gets the categorical help block.
        if isinstance(e, StrataConfigError) and "not found" in str(e):
            click.echo(_format_categorical_help(dataset_zarr))
        sys.exit(1)
