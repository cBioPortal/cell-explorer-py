"""Tests for StrataZarrAccess adapter."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from cell_explorer_agent.tools.strata_protocol import (
    CoarseStrataResult,
    StrataAccess,
)
from cell_explorer_api.services.zarr_adapter import StrataZarrAccess


def _make_fake_strata_store(
    *,
    coarse_slugs: list[str] | None = None,
    coarse_axes_map: dict[str, list[str]] | None = None,
    coarse_tables: dict | None = None,
) -> MagicMock:
    store = MagicMock()
    slugs = list(coarse_slugs or ["cell_type"])
    axes_map = coarse_axes_map or {s: [s] for s in slugs}
    tables = coarse_tables or {}
    store.coarse_slugs.return_value = slugs
    store.coarse_axes.side_effect = lambda s: axes_map[s]

    async def _read(slug: str):
        return tables[slug]

    store.read_coarse = AsyncMock(side_effect=_read)
    return store


def test_satisfies_strata_access_protocol():
    adapter = StrataZarrAccess(_make_fake_strata_store())
    assert isinstance(adapter, StrataAccess)


def test_coarse_slugs_delegates_to_store():
    store = _make_fake_strata_store(coarse_slugs=["cell_type", "donor"])
    adapter = StrataZarrAccess(store)
    assert adapter.coarse_slugs() == ["cell_type", "donor"]


def test_coarse_axes_delegates_to_store():
    store = _make_fake_strata_store(
        coarse_slugs=["cell_type"],
        coarse_axes_map={"cell_type": ["cell_type"]},
    )
    adapter = StrataZarrAccess(store)
    assert adapter.coarse_axes("cell_type") == ["cell_type"]


@pytest.mark.asyncio
async def test_read_coarse_converts_to_wire_format():
    """The adapter constructs a CoarseStrataResult from the underlying table."""
    # Build a mock CoarseStrataTable-like object (just needs the right attrs).
    mock_table = MagicMock()
    mock_table.axes = ["cell_type"]
    mock_table.stratum_keys = np.array([["A"], ["B"]])
    mock_table.sum_x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mock_table.sum_xx = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
    mock_table.nnz = np.array([[1, 1], [1, 1]], dtype=np.int32)
    mock_table.n_cells = np.array([2, 2], dtype=np.int32)

    store = _make_fake_strata_store(
        coarse_tables={"cell_type": mock_table},
    )
    adapter = StrataZarrAccess(store)
    result = await adapter.read_coarse("cell_type")

    assert isinstance(result, CoarseStrataResult)
    assert result.axes == ["cell_type"]
    assert result.stratum_keys.shape == (2, 1)
    np.testing.assert_array_equal(result.sum_x, mock_table.sum_x)
    np.testing.assert_array_equal(result.n_cells, mock_table.n_cells)
