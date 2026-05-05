"""Tests for cell_explorer_agent.tools.parallel.reduce_gene_columns."""

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from cell_explorer_agent.tools.parallel import reduce_gene_columns
from cell_explorer_agent.tools.zarr_protocol import ObsColumn, ObsColumnSpec


@dataclass
class _CountingZarr:
    """Test double — records gene_column calls and tracks concurrent in-flight count."""

    expression: dict[str, np.ndarray]
    delay_s: float = 0.005
    fail_on: str | None = None
    _active: int = 0
    _peak: int = 0

    @property
    def peak_concurrency(self) -> int:
        return self._peak

    async def shape(self) -> tuple[int, int]:
        return (10, len(self.expression))

    async def attrs(self) -> dict:
        return {}

    async def obsm_keys(self) -> list[str]:
        return []

    async def var_names(self) -> list[str]:
        return list(self.expression.keys())

    async def obs_columns(self) -> list[ObsColumnSpec]:
        return []

    async def obs_column(self, name: str) -> ObsColumn:
        raise KeyError(name)

    async def gene_index(self, gene: str) -> int:
        if gene not in self.expression:
            raise KeyError(gene)
        return list(self.expression).index(gene)

    async def gene_column(self, gene: str) -> np.ndarray:
        if gene == self.fail_on:
            raise RuntimeError(f"boom for {gene}")
        self._active += 1
        self._peak = max(self._peak, self._active)
        try:
            await asyncio.sleep(self.delay_s)
            if gene not in self.expression:
                raise KeyError(gene)
            return self.expression[gene]
        finally:
            self._active -= 1

    async def obs_mask(self, obs_col: str, value: str) -> np.ndarray:
        raise KeyError(obs_col)


def _make_zarr(genes: list[str], **kwargs) -> _CountingZarr:
    expression = {g: np.array([float(i)] * 5, dtype="float32") for i, g in enumerate(genes)}
    return _CountingZarr(expression=expression, **kwargs)


@pytest.mark.asyncio
async def test_alignment_preserves_input_order():
    """Results are returned in the same order as the input gene list."""
    z = _make_zarr(["g1", "g2", "g3"])

    def reducer(gene: str, expr: np.ndarray) -> tuple[str, float]:
        return (gene, float(expr.mean()))

    out = await reduce_gene_columns(z, ["g1", "g2", "g3"], reducer, max_concurrency=2)

    assert out == [("g1", 0.0), ("g2", 1.0), ("g3", 2.0)]


@pytest.mark.asyncio
async def test_empty_input_returns_empty_list():
    """Empty gene list returns [] without touching the store."""
    z = _make_zarr(["g1"])

    def reducer(gene: str, expr: np.ndarray) -> int:
        return len(expr)

    out = await reduce_gene_columns(z, [], reducer, max_concurrency=4)
    assert out == []
    assert z.peak_concurrency == 0  # nothing was fetched


@pytest.mark.asyncio
async def test_concurrency_cap_enforced():
    """Peak in-flight gene_column calls never exceeds max_concurrency."""
    genes = [f"g{i}" for i in range(20)]
    z = _make_zarr(genes, delay_s=0.01)

    def reducer(gene: str, expr: np.ndarray) -> str:
        return gene

    await reduce_gene_columns(z, genes, reducer, max_concurrency=4)

    assert z.peak_concurrency <= 4
    assert z.peak_concurrency > 1, "should have been actually concurrent"


@pytest.mark.asyncio
async def test_error_propagates_and_cancels_others():
    """Any single gene_column error fails the whole call."""
    genes = [f"g{i}" for i in range(10)]
    z = _make_zarr(genes, delay_s=0.005, fail_on="g5")

    def reducer(gene: str, expr: np.ndarray) -> str:
        return gene

    with pytest.raises(RuntimeError, match="boom for g5"):
        await reduce_gene_columns(z, genes, reducer, max_concurrency=3)


@pytest.mark.asyncio
async def test_reducer_receives_gene_name_and_array():
    """Reducer signature is (gene_name, expr) — both are passed."""
    z = _make_zarr(["alpha", "beta"])

    seen: list[tuple[str, int]] = []

    def reducer(gene: str, expr: np.ndarray) -> None:
        seen.append((gene, expr.size))

    await reduce_gene_columns(z, ["alpha", "beta"], reducer, max_concurrency=2)

    # Order of `seen` is not guaranteed (concurrent), but contents are.
    assert sorted(seen) == [("alpha", 5), ("beta", 5)]
