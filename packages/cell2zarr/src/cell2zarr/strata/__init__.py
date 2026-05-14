"""Strata tables: precomputed per-(stratum, gene) sums for fast aggregate queries.

See `cell2zarr build-strata --help` for the CLI. See docs/superpowers/specs/
2026-05-14-strata-build-design.md for the design.
"""

from cell2zarr.strata.exceptions import (
    StrataError,
    StrataConfigError,
    StrataExistsError,
    StrataPreflightError,
    StrataBuildError,
    StrataWriteError,
)

__all__ = [
    "StrataError",
    "StrataConfigError",
    "StrataExistsError",
    "StrataPreflightError",
    "StrataBuildError",
    "StrataWriteError",
]
