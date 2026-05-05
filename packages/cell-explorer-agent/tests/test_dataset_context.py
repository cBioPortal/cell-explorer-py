"""Tests for build_dataset_context(), focusing on ObsColumnInfo.values population."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from cell_explorer_agent.prompt.dataset_context import ObsColumnInfo, build_dataset_context
from cell_explorer_agent.tools.zarr_protocol import ObsColumn, ObsColumnSpec


@dataclass
class _StubZarr:
    """Minimal ZarrAccess stub for dataset_context tests."""

    _specs: list[ObsColumnSpec] = field(default_factory=list)
    _n_obs: int = 100
    _n_var: int = 50

    async def shape(self) -> tuple[int, int]:
        return (self._n_obs, self._n_var)

    async def attrs(self) -> dict:
        return {}

    async def obs_columns(self) -> list[ObsColumnSpec]:
        return list(self._specs)

    async def obs_column(self, name: str) -> ObsColumn:
        raise KeyError(name)

    async def var_names(self) -> list[str]:
        return []

    async def obsm_keys(self) -> list[str]:
        return []

    async def gene_index(self, gene: str) -> int:
        raise KeyError(gene)

    async def gene_column(self, gene: str) -> np.ndarray:
        raise KeyError(gene)

    async def obs_mask(self, obs_col: str, value: str) -> np.ndarray:
        raise KeyError(obs_col)


@pytest.mark.asyncio
async def test_values_populated_for_low_cardinality_categorical():
    """Categorical column with cardinality <= 50 gets values populated."""
    cats = ["T cell", "B cell", "Monocyte"]
    z = _StubZarr(
        _specs=[
            ObsColumnSpec(
                name="cell_type",
                dtype="categorical",
                cardinality=3,
                categories=cats,
            )
        ]
    )
    ctx = await build_dataset_context(z, slug="ds1", name="Test", description="")
    col = ctx.obs_columns[0]
    assert isinstance(col, ObsColumnInfo)
    assert col.values == cats


@pytest.mark.asyncio
async def test_values_none_for_high_cardinality_categorical():
    """Categorical column with cardinality > 50 gets values=None (cap enforced)."""
    cats = [f"cat_{i}" for i in range(60)]
    z = _StubZarr(
        _specs=[
            ObsColumnSpec(
                name="sample_id",
                dtype="categorical",
                cardinality=60,
                categories=cats,
            )
        ]
    )
    ctx = await build_dataset_context(z, slug="ds2", name="Test", description="")
    col = ctx.obs_columns[0]
    assert col.values is None


@pytest.mark.asyncio
async def test_values_none_for_numeric_column():
    """Numeric column always gets values=None."""
    z = _StubZarr(
        _specs=[
            ObsColumnSpec(
                name="n_counts",
                dtype="numeric",
                cardinality=None,
                categories=None,
            )
        ]
    )
    ctx = await build_dataset_context(z, slug="ds3", name="Test", description="")
    col = ctx.obs_columns[0]
    assert col.values is None
