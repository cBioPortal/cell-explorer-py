import socket
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
import zarr
from aiohttp import web
from fastapi.testclient import TestClient


@pytest.fixture()
def static_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a minimal index.html."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><body>app</body></html>")
    assets = dist / "assets"
    assets.mkdir()
    (assets / "main.js").write_text("console.log('hello')")
    return dist


@pytest.fixture()
def client() -> TestClient:
    """Test client with no static serving (API-only mode)."""
    from cell_explorer_api.main import create_app

    app = create_app()
    return TestClient(app)


# --- Fixtures serving generated zarr stores over HTTP -----------------------
#
# Mirrors packages/zarr-access/tests/conftest.py, but generates tiny stores at
# test time rather than shipping fixtures — the error cases (non-AnnData store,
# absent X) have no checked-in equivalent, and a 12x5 matrix keeps the suite fast.


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _tiny_adata() -> "ad.AnnData":
    n_obs, n_vars = 12, 5
    X = np.arange(n_obs * n_vars, dtype="float32").reshape(n_obs, n_vars)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["T"] * 6 + ["B"] * 6),
            "n_counts": np.arange(n_obs, dtype="float32"),
        },
        index=[f"cell{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"feature_name": [f"GENE{i}" for i in range(n_vars)]},
        index=[f"ENSG{i:05d}" for i in range(n_vars)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = np.zeros((n_obs, 2), dtype="float32")
    adata.layers["counts"] = X.copy()
    return adata


@pytest.fixture(scope="session")
def zarr_fixtures(tmp_path_factory) -> Path:
    """Generate v2 + v3 AnnData stores and two malformed stores."""
    root = tmp_path_factory.mktemp("zarr-fixtures")

    # v3 — what production writes.
    ad.settings.zarr_write_format = 3
    _tiny_adata().write_zarr(str(root / "tiny_v3.zarr"))

    # v2 — consolidate explicitly so .zmetadata exists, matching real v2 stores.
    ad.settings.zarr_write_format = 2
    _tiny_adata().write_zarr(str(root / "tiny_v2.zarr"))
    zarr.consolidate_metadata(str(root / "tiny_v2.zarr"))

    # Restore the default so later tests in the session are unaffected.
    ad.settings.zarr_write_format = 3

    # Not an AnnData store at all: a bare group with one array.
    plain = zarr.open_group(str(root / "not_anndata.zarr"), mode="w", zarr_format=3)
    plain.create_array("values", shape=(4,), dtype="float32")

    # Claims to be AnnData but has no X.
    headless = zarr.open_group(str(root / "no_x.zarr"), mode="w", zarr_format=3)
    headless.attrs["encoding-type"] = "anndata"
    headless.attrs["encoding-version"] = "0.1.0"

    return root


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def zarr_server(zarr_fixtures):
    """Serve the generated stores over HTTP on a random port.

    This fixture is pinned to a session-scoped event loop. Any test module
    that consumes it (directly or via a fixture that depends on it) MUST set
    `pytestmark = pytest.mark.asyncio(loop_scope="session")` at module level —
    the root pyproject.toml defaults async test loops to function scope, and
    without that marker the test run deadlocks waiting on this fixture, with
    no error raised.
    """
    app = web.Application()
    app.router.add_static("/", zarr_fixtures, show_index=True)

    port = _find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()
