from pathlib import Path

from fastapi.testclient import TestClient

from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


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
