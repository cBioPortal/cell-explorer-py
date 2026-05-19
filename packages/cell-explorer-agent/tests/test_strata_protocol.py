"""Tests for StrataAccess Protocol and CoarseStrataResult dataclass."""

import numpy as np

from cell_explorer_agent.tools.strata_protocol import (
    CoarseStrataResult,
    StrataAccess,
)


def test_coarse_strata_result_constructs_with_required_fields():
    """CoarseStrataResult is a frozen dataclass holding the wire-format arrays."""
    result = CoarseStrataResult(
        axes=["cell_type"],
        stratum_keys=np.array([["A"], ["B"]]),
        sum_x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        sum_xx=np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32),
        nnz=np.array([[1, 1], [1, 1]], dtype=np.int32),
        n_cells=np.array([2, 2], dtype=np.int32),
    )
    assert result.axes == ["cell_type"]
    assert result.stratum_keys.shape == (2, 1)
    assert result.sum_x.shape == (2, 2)


def test_coarse_strata_result_is_frozen():
    """Tables should be immutable values — assignment raises."""
    import dataclasses
    result = CoarseStrataResult(
        axes=["cell_type"],
        stratum_keys=np.array([["A"]]),
        sum_x=np.zeros((1, 1), dtype=np.float32),
        sum_xx=np.zeros((1, 1), dtype=np.float32),
        nnz=np.zeros((1, 1), dtype=np.int32),
        n_cells=np.array([0], dtype=np.int32),
    )
    try:
        result.axes = ["other"]  # type: ignore
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("expected FrozenInstanceError")


def test_strata_access_protocol_is_runtime_checkable():
    """isinstance check works for runtime Protocol verification."""
    from cell_explorer_agent.tools.strata_protocol import AtomicStrataResult

    class _Fake:
        def coarse_slugs(self) -> list[str]:
            return []

        def coarse_axes(self, slug: str) -> list[str]:
            return []

        async def read_coarse(self, slug: str) -> CoarseStrataResult:
            raise NotImplementedError

        def has_atomic(self) -> bool:
            return False

        def atomic_axes(self) -> list[str] | None:
            return None

        async def read_atomic(self) -> AtomicStrataResult:
            raise NotImplementedError

    assert isinstance(_Fake(), StrataAccess)
