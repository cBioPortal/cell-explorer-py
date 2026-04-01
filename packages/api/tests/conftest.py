from pathlib import Path

import pytest
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
