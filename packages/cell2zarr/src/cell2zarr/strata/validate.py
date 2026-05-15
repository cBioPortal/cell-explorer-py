"""Preflight validation for strata build.

All checks raise StrataConfigError (caused by user input) or
StrataPreflightError (consequence of valid config but dataset can't
support it) before any X read.
"""
import numpy as np
import zarr

from cell2zarr.strata.exceptions import (
    StrataConfigError,
    StrataPreflightError,
)
from cell2zarr.strata.io import read_obs_categorical

NULLISH_THRESHOLD = 0.05  # 5% null tolerated, anything above -> error


def list_categorical_obs_columns(root: zarr.Group) -> dict[str, int]:
    """Enumerate categorical-shaped obs columns and their cardinalities.

    Used to render the helpful error message when --atomic-axes is missing.
    """
    if "obs" not in root:
        return {}
    out: dict[str, int] = {}
    obs = root["obs"]
    for name in obs:
        if not isinstance(obs[name], zarr.Group):
            continue
        if "codes" in obs[name] and "categories" in obs[name]:
            out[name] = len(np.asarray(obs[name]["categories"]))
    return dict(sorted(out.items()))


def validate_obs_columns(root: zarr.Group, axis_names: list[str]) -> None:
    """Confirm each axis name maps to a categorical obs column with low null-rate.

    Raises StrataConfigError on any failure with a message that names the column.
    """
    if "obs" not in root:
        raise StrataConfigError("dataset has no obs/ group")
    obs = root["obs"]

    for name in axis_names:
        if name not in obs:
            raise StrataConfigError(f"obs column '{name}' not found")
        col = obs[name]
        is_categorical = (
            isinstance(col, zarr.Group)
            and "codes" in col
            and "categories" in col
        )
        if not is_categorical:
            raise StrataConfigError(
                f"obs column '{name}' is numeric (not categorical). "
                f"Atomic axes must be categorical."
            )

        # Null-rate check: AnnData encodes missing categorical values as code = -1
        codes = np.asarray(col["codes"])
        null_count = int((codes < 0).sum())
        if codes.size > 0:
            null_rate = null_count / codes.size
            if null_rate > NULLISH_THRESHOLD:
                raise StrataConfigError(
                    f"obs column '{name}' has {null_rate:.1%} missing values "
                    f"(threshold {NULLISH_THRESHOLD:.0%})"
                )


def estimate_atomic_cardinality(
    root: zarr.Group,
    axis_names: list[str],
    max_cardinality: int | None = None,
) -> int:
    """Count unique tuples on the named axes.

    If max_cardinality is set and the count exceeds it, raise StrataPreflightError
    with a message that recommends dropping a fine-grained axis or raising the cap.
    """
    if not axis_names:
        return 0
    columns = [read_obs_categorical(root, name) for name in axis_names]
    # Pack the codes from each axis into a single key per cell
    code_matrix = np.column_stack([c.codes.astype(np.int64) for c in columns])
    unique_rows = np.unique(code_matrix, axis=0)
    n_atomic = len(unique_rows)

    if max_cardinality is not None and n_atomic > max_cardinality:
        raise StrataPreflightError(
            f"Atomic strata count {n_atomic} exceeds limit {max_cardinality}. "
            f"Drop a fine-grained axis, or raise --max-atomic-cardinality."
        )
    return n_atomic
