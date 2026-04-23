"""pytest fixtures shared across all agent tests."""

import sys
from pathlib import Path

import pytest

# Ensure the package root is on sys.path so `tests.fakes.*` is importable
# when pytest is invoked from the workspace root (which uses importlib mode
# but only processes workspace-level pythonpath entries, not per-package ones).
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from tests.fakes.fake_zarr import FakeZarrAccess


@pytest.fixture
def fake_zarr() -> FakeZarrAccess:
    return FakeZarrAccess.default()
