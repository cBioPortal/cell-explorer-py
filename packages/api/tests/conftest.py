from pathlib import Path

import pytest


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
