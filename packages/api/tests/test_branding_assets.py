import json
from pathlib import Path

from fastapi.testclient import TestClient

from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


def _brand_dir(tmp_path: Path) -> Path:
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "BTC"}))
    (brand_dir / "logo-white.svg").write_text("<svg id='btc'/>")
    return brand_dir


def test_brand_asset_is_served(tmp_path: Path):
    app = create_app(Settings(brand_dir=_brand_dir(tmp_path)))
    client = TestClient(app)
    response = client.get("/brand/logo-white.svg")
    assert response.status_code == 200
    assert "btc" in response.text


def test_missing_brand_asset_404s(tmp_path: Path):
    app = create_app(Settings(brand_dir=_brand_dir(tmp_path)))
    client = TestClient(app)
    assert client.get("/brand/absent.svg").status_code == 404


def test_brand_route_absent_when_brand_dir_unset(static_dir: Path):
    # With static serving on and no brand, /brand/* falls through to the SPA
    # catch-all and returns index.html rather than a brand asset.
    app = create_app(Settings(static_dir=static_dir))
    client = TestClient(app)
    response = client.get("/brand/logo-white.svg")
    assert "<!doctype html>" in response.text


def test_brand_assets_take_precedence_over_spa_catchall(tmp_path: Path, static_dir: Path):
    app = create_app(Settings(static_dir=static_dir, brand_dir=_brand_dir(tmp_path)))
    client = TestClient(app)
    response = client.get("/brand/logo-white.svg")
    assert response.status_code == 200
    assert "btc" in response.text


def test_api_routes_still_take_precedence(tmp_path: Path):
    app = create_app(Settings(brand_dir=_brand_dir(tmp_path)))
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}


def test_traversal_outside_brand_dir_is_refused(tmp_path: Path):
    # Strengthened over the brief's version: TestClient normalizes a plain
    # "../" client-side before the request is even sent, so that form alone
    # never reaches the server-side guard and a disjunctive assertion would
    # pass regardless of whether StaticFiles' traversal protection works. We
    # additionally send a URL-encoded variant ("%2e%2e"), which survives
    # client-side normalization and does exercise the guard, and we assert a
    # specific outcome (not 200, and the secret's contents absent) for both
    # forms instead of an either/or.
    brand_dir = _brand_dir(tmp_path)
    (tmp_path / "secret.txt").write_text("do not serve me")

    app = create_app(Settings(brand_dir=brand_dir))
    client = TestClient(app)

    response = client.get("/brand/../secret.txt")
    assert response.status_code != 200
    assert "do not serve me" not in response.text

    response = client.get("/brand/%2e%2e/secret.txt")
    assert response.status_code != 200
    assert "do not serve me" not in response.text
