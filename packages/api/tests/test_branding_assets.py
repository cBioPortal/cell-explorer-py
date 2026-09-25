import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


def _brand_dir(tmp_path: Path) -> Path:
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    # The logo is *referenced*, not merely present: only filenames brand.json
    # resolves onto are served.
    (brand_dir / "brand.json").write_text(
        json.dumps({"name": "BTC", "logo": {"onDark": "logo-white.svg"}})
    )
    (brand_dir / "logo-white.svg").write_text("<svg id='btc'/>")
    return brand_dir


def test_brand_asset_is_served(tmp_path: Path):
    app = create_app(Settings(brand_dir=_brand_dir(tmp_path)))
    client = TestClient(app)
    response = client.get("/brand/logo-white.svg")
    assert response.status_code == 200
    assert "btc" in response.text


def test_brand_asset_answers_head(tmp_path: Path):
    # The StaticFiles mount this route replaced answered HEAD; a bare FastAPI
    # @app.get does not. CDNs and health checks issue HEAD even though browsers
    # fetch images with GET, so losing it is a real regression.
    app = create_app(Settings(brand_dir=_brand_dir(tmp_path)))
    client = TestClient(app)
    response = client.head("/brand/logo-white.svg")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=300"
    assert response.content == b""


def test_head_on_an_unreferenced_file_404s(tmp_path: Path):
    app = create_app(Settings(brand_dir=_brand_dir(tmp_path)))
    client = TestClient(app)
    assert client.head("/brand/brand.json").status_code == 404


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


# --- Only referenced assets are served ------------------------------------


def test_unreferenced_file_in_brand_dir_is_not_served(tmp_path: Path):
    """BRAND_DIR is an operator directory; only what brand.json points at is web-reachable."""
    brand_dir = _brand_dir(tmp_path)
    (brand_dir / "notes.txt").write_text("internal planning notes")
    (brand_dir / "draft.svg").write_text("<svg id='unshipped'/>")

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))

    # Referenced: served.
    assert client.get("/brand/logo-white.svg").status_code == 200

    # Present but unreferenced: not served, whatever the extension.
    for name, body in (("notes.txt", "internal"), ("draft.svg", "unshipped")):
        response = client.get(f"/brand/{name}")
        assert response.status_code == 404, name
        assert body not in response.text


def test_brand_json_itself_is_not_served(tmp_path: Path):
    """The config is input to the server, not a public document."""
    client = TestClient(create_app(Settings(brand_dir=_brand_dir(tmp_path))))

    response = client.get("/brand/brand.json")
    assert response.status_code == 404
    assert "logo-white.svg" not in response.text


def test_asset_referenced_but_missing_on_disk_is_not_served(tmp_path: Path):
    """A filename dropped by the existence check never enters the served set."""
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps({"name": "BTC", "favicon": {"ico": "absent.ico"}})
    )

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))
    assert client.get("/brand/absent.ico").status_code == 404


def test_symlink_out_of_brand_dir_is_refused(tmp_path: Path):
    """A referenced name may still be a symlink pointing out of the bundle."""
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    secret = tmp_path / "secret.svg"
    secret.write_text("<svg id='do-not-serve-me'/>")
    (brand_dir / "logo.svg").symlink_to(secret)
    (brand_dir / "brand.json").write_text(
        json.dumps({"name": "BTC", "logo": {"onDark": "logo.svg"}})
    )

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))

    response = client.get("/brand/logo.svg")
    assert response.status_code == 404
    assert "do-not-serve-me" not in response.text


# --- Caching --------------------------------------------------------------


def test_brand_asset_carries_a_short_cache_control(tmp_path: Path):
    """Distinct from /assets/* — those are content-hashed, these are not."""
    client = TestClient(create_app(Settings(brand_dir=_brand_dir(tmp_path))))

    response = client.get("/brand/logo-white.svg")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=300"


# --- Percent-encoded filenames --------------------------------------------


@pytest.mark.parametrize("filename", ["logo one.svg", "logo#1.svg", "100%.svg"])
def test_awkward_filenames_round_trip_through_the_generated_url(
    tmp_path: Path, filename: str
):
    """The URL /api/info advertises must be the URL that actually fetches the file."""
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps({"name": "BTC", "logo": {"onDark": filename}})
    )
    (brand_dir / filename).write_text("<svg id='btc'/>")

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))

    url = client.get("/api/info").json()["brand"]["logo_on_dark"]
    assert url is not None
    # Percent-encoded, so no character in the name is reparsed as URL syntax.
    assert filename not in url
    response = client.get(url)
    assert response.status_code == 200
    assert "btc" in response.text


# --- An unreadable BRAND_DIR must not stop the app booting -----------------


def test_unstattable_brand_dir_does_not_stop_create_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
):
    """The outage case: a non-root container with a root-owned mount parent.

    Path.is_dir() swallows ENOENT/ENOTDIR but re-raises EACCES, so any second
    stat of BRAND_DIR outside load_bundle's guard takes create_app down. Forced
    deterministically rather than via chmod, because CI runs as root, where
    permission bits are bypassed and a chmod-based test asserts nothing.
    """
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "BTC"}))

    real_is_dir = Path.is_dir

    def is_dir(self: Path, *args, **kwargs):
        if self == brand_dir:
            raise PermissionError(13, "Permission denied")
        return real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", is_dir)

    with caplog.at_level("WARNING", logger="cell_explorer_api.branding"):
        app = create_app(Settings(brand_dir=brand_dir))

    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    # Degraded to unbranded, and loudly.
    assert client.get("/api/info").json()["brand"] is None
    assert any("Could not stat BRAND_DIR" in r.getMessage() for r in caplog.records)


def test_unstattable_brand_dir_registers_no_brand_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, static_dir: Path
):
    """The mount decision follows the load decision, so /brand/* falls through."""
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "BTC"}))

    real_is_dir = Path.is_dir

    def is_dir(self: Path, *args, **kwargs):
        if self == brand_dir:
            raise PermissionError(13, "Permission denied")
        return real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", is_dir)

    client = TestClient(create_app(Settings(static_dir=static_dir, brand_dir=brand_dir)))
    assert "<!doctype html>" in client.get("/brand/logo-white.svg").text


def test_overlong_asset_filename_does_not_stop_create_app(tmp_path: Path):
    """The invariant is stated at create_app, so it is asserted at create_app.

    No monkeypatching: an unbounded filename in brand.json is enough on its own,
    because the OS refuses to answer whether a >255-byte name exists rather than
    reporting absence, and that OSError used to escape the factory.
    """
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps(
            {
                "name": "BTC",
                "logo": {"onDark": "a" * 5000 + ".svg", "onLight": "logo.svg"},
            }
        )
    )
    (brand_dir / "logo.svg").write_text("<svg id='btc'/>")

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))

    # Booted, branded, and only the one bad field was dropped.
    brand = client.get("/api/info").json()["brand"]
    assert brand["name"] == "BTC"
    assert brand["logo_on_dark"] is None
    assert brand["logo_on_light"] == "/brand/logo.svg"
    assert client.get("/brand/logo.svg").status_code == 200
