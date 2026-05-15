"""Pydantic config model for cell2zarr.strata.build_strata."""
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator


class StrataConfig(BaseModel):
    """Configuration for a single `cell2zarr build-strata` invocation.

    Populated either from CLI flags or from a YAML config file (via from_yaml).
    """

    atomic_axes: list[str] = Field(
        min_length=1,
        description="obs/ column names that define atomic strata. Order is preserved.",
    )
    coarse: list[list[str]] = Field(
        default_factory=list,
        description="Each entry is a list of axes (subset of atomic_axes) defining a coarse table.",
    )
    max_atomic_cardinality: int = Field(
        default=50_000,
        ge=1,
        description="Reject the build if estimated |atomic strata| exceeds this.",
    )
    force: bool = Field(
        default=False,
        description="If true, overwrite an existing uns/strata/ group.",
    )

    @model_validator(mode="after")
    def _coarse_axes_subset_of_atomic(self) -> Self:
        atomic = set(self.atomic_axes)
        for cols in self.coarse:
            extra = set(cols) - atomic
            if extra:
                raise ValueError(
                    f"--coarse axes {sorted(extra)} are not in --atomic-axes {self.atomic_axes}"
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load a StrataConfig from a YAML file.

        Normalizes single-string entries in `coarse` to single-element lists so the
        internal shape is always list[list[str]].
        """
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        # Normalize coarse: each entry may be a string (single-axis) or a list (multi-axis)
        data["coarse"] = [
            [c] if isinstance(c, str) else list(c) for c in data.get("coarse", [])
        ]
        return cls.model_validate(data)
