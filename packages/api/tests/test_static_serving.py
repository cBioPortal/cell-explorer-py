from pathlib import Path

from fastapi.testclient import TestClient

from cell_explorer_api.config import Settings
from cell_explorer_api.main import _resolve_static_file, create_app


def test_spa_catchall_returns_index_html(static_dir: Path):
    app = create_app(Settings(static_dir=static_dir))
    client = TestClient(app)
    response = client.get("/some/frontend/route")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text


def test_static_assets_served(static_dir: Path):
    app = create_app(Settings(static_dir=static_dir))
    client = TestClient(app)
    response = client.get("/assets/main.js")
    assert response.status_code == 200
    assert "console.log" in response.text


def test_api_routes_take_precedence_over_catchall(static_dir: Path):
    app = create_app(Settings(static_dir=static_dir))
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_no_static_dir_returns_api_only():
    app = create_app(Settings(static_dir=None))
    client = TestClient(app)
    # API still works
    response = client.get("/api/health")
    assert response.status_code == 200
    # Non-API routes get 404 (no static serving)
    response = client.get("/some/route")
    assert response.status_code == 404


def test_invalid_static_dir_returns_503(tmp_path: Path):
    app = create_app(Settings(static_dir=tmp_path / "nonexistent"))
    client = TestClient(app)
    response = client.get("/some/route")
    assert response.status_code == 503
    assert "Frontend not found" in response.json()["detail"]


def test_root_public_file_served_not_index_html(static_dir: Path):
    """Vite copies public/ to the dist root — favicons must not hit the SPA fallback."""
    (static_dir / "favicon.svg").write_text("<svg/>")
    client = TestClient(create_app(Settings(static_dir=static_dir)))
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert response.text == "<svg/>"
    assert "text/html" not in response.headers["content-type"]


def test_nested_public_file_served(static_dir: Path):
    nested = static_dir / "icons"
    nested.mkdir()
    (nested / "logo.png").write_bytes(b"\x89PNG")
    client = TestClient(create_app(Settings(static_dir=static_dir)))
    response = client.get("/icons/logo.png")
    assert response.status_code == 200
    assert response.content == b"\x89PNG"


def test_missing_root_file_still_falls_back_to_index_html(static_dir: Path):
    client = TestClient(create_app(Settings(static_dir=static_dir)))
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text


def test_api_precedence_unaffected_by_real_file(static_dir: Path):
    """A file named like an API path must not shadow the 404 contract."""
    client = TestClient(create_app(Settings(static_dir=static_dir)))
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404


class TestResolveStaticFile:
    def test_serves_existing_file(self, static_dir: Path):
        assert _resolve_static_file(static_dir, "index.html") == (
            static_dir / "index.html"
        ).resolve()

    def test_rejects_traversal_outside_root(self, static_dir: Path):
        (static_dir.parent / "secret.txt").write_text("nope")
        assert _resolve_static_file(static_dir, "../secret.txt") is None

    def test_rejects_directory(self, static_dir: Path):
        assert _resolve_static_file(static_dir, "assets") is None

    def test_rejects_empty_and_missing(self, static_dir: Path):
        assert _resolve_static_file(static_dir, "") is None
        assert _resolve_static_file(static_dir, "nope.png") is None
