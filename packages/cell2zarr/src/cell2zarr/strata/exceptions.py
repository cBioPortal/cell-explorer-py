"""Exception hierarchy for cell2zarr.strata."""


class StrataError(Exception):
    """Base class for all strata-related errors."""


class StrataConfigError(StrataError):
    """Bad configuration — missing axes, malformed YAML, axes that don't subset, etc."""


class StrataExistsError(StrataError):
    """uns/strata/atomic already exists and --force was not passed."""


class StrataPreflightError(StrataError):
    """Pre-flight check failed — bad obs column, cardinality cap, missing X."""


class StrataBuildError(StrataError):
    """Atomic build failed mid-run — OOM, zarr-read failure, etc."""


class StrataWriteError(StrataError):
    """Writing the resulting strata tables back to zarr failed."""
