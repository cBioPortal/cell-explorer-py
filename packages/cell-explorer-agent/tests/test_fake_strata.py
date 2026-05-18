"""Smoke test that FakeStrataAccess satisfies the StrataAccess Protocol."""

import numpy as np
import pytest

from cell_explorer_agent.tools.strata_protocol import StrataAccess
from tests.fakes.fake_strata import FakeStrataAccess


def test_fake_strata_satisfies_protocol():
    fake = FakeStrataAccess.default()
    assert isinstance(fake, StrataAccess)


def test_fake_strata_has_default_coarse_slugs():
    fake = FakeStrataAccess.default()
    assert fake.coarse_slugs() == ["cell_type"]
    assert fake.coarse_axes("cell_type") == ["cell_type"]


async def test_fake_strata_read_coarse_returns_expected_shape():
    fake = FakeStrataAccess.default()
    coarse = await fake.read_coarse("cell_type")
    # 3 cell types × 50 genes (matches FakeZarrAccess default)
    assert coarse.axes == ["cell_type"]
    assert coarse.stratum_keys.shape == (3, 1)
    assert coarse.sum_x.shape == (3, 50)
    assert coarse.n_cells.shape == (3,)
    assert int(coarse.n_cells.sum()) == 100


async def test_fake_strata_read_coarse_unknown_slug_raises():
    fake = FakeStrataAccess.default()
    with pytest.raises(KeyError):
        await fake.read_coarse("no_such_slug")
