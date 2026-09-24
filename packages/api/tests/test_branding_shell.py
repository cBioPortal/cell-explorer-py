import json
from pathlib import Path

from fastapi.testclient import TestClient

from cell_explorer_api.branding import DEFAULT_BRAND, parse_brand
from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app
from cell_explorer_api.shell import render_index_html, render_webmanifest


def _brand(raw: dict):
    return parse_brand(raw, warn=lambda _msg: None)


def _read(shell_dir: Path, name: str) -> str:
    return (shell_dir / name).read_text()


def test_default_brand_leaves_html_untouched(shell_dir: Path):
    original = _read(shell_dir, "index.html")
    assert render_index_html(original, DEFAULT_BRAND) == original


def test_title_is_replaced(shell_dir: Path):
    brand = _brand({"title": "Break Through Cancer — Cell Explorer"})
    out = render_index_html(_read(shell_dir, "index.html"), brand)
    assert "<title>Break Through Cancer — Cell Explorer</title>" in out
    assert "cBioPortal Cell Explorer" not in out


def test_theme_color_is_replaced(shell_dir: Path):
    brand = _brand({"colors": {"ink": "#240D00"}})
    out = render_index_html(_read(shell_dir, "index.html"), brand)
    assert '<meta name="theme-color" content="#240D00" />' in out


def test_favicon_links_point_at_brand_assets(shell_dir: Path):
    brand = _brand(
        {"favicon": {"ico": "btc.ico", "svg": "btc.svg", "appleTouch": "btc-180.png"}}
    )
    out = render_index_html(_read(shell_dir, "index.html"), brand)
    assert 'href="/brand/btc.ico"' in out
    assert 'href="/brand/btc.svg"' in out
    assert 'href="/brand/btc-180.png"' in out


def test_unset_favicon_fields_keep_their_defaults(shell_dir: Path):
    brand = _brand({"favicon": {"ico": "btc.ico"}})
    out = render_index_html(_read(shell_dir, "index.html"), brand)
    assert 'href="/brand/btc.ico"' in out
    assert 'href="/favicon.svg"' in out  # untouched


def test_spa_redirect_script_survives_byte_identical(shell_dir: Path):
    # An HTML parse-and-reserialize would put this at risk. Anchored
    # substitution must leave it exactly as authored.
    original = _read(shell_dir, "index.html")
    script = original[original.index("<script>") : original.index("</script>")]
    brand = _brand({"name": "BTC", "colors": {"ink": "#240D00"}})
    out = render_index_html(original, brand)
    assert script in out


def test_brand_text_is_html_escaped(shell_dir: Path):
    brand = _brand({"title": "Acme <script>alert(1)</script>"})
    out = render_index_html(_read(shell_dir, "index.html"), brand)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_webmanifest_fields_replaced(shell_dir: Path):
    brand = _brand(
        {
            "name": "Break Through Cancer",
            "shortName": "BTC",
            "title": "BTC — Cell Explorer",
            "colors": {"ink": "#240D00"},
            "favicon": {"png192": "btc-192.png", "png512": "btc-512.png"},
        }
    )
    out = json.loads(render_webmanifest(_read(shell_dir, "site.webmanifest"), brand))
    assert out["name"] == "BTC — Cell Explorer"
    assert out["short_name"] == "BTC"
    assert out["background_color"] == "#240D00"
    assert out["theme_color"] == "#240D00"
    assert out["icons"][0]["src"] == "/brand/btc-192.png"
    assert out["icons"][1]["src"] == "/brand/btc-512.png"


def test_webmanifest_untouched_for_default_brand(shell_dir: Path):
    original = _read(shell_dir, "site.webmanifest")
    assert render_webmanifest(original, DEFAULT_BRAND) == original


def test_served_index_html_carries_brand(tmp_path: Path, shell_dir: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps({"name": "BTC", "favicon": {"ico": "btc.ico"}})
    )
    (brand_dir / "btc.ico").write_bytes(b"\x00")

    app = create_app(Settings(static_dir=shell_dir, brand_dir=brand_dir))
    client = TestClient(app)

    response = client.get("/some/spa/route")
    assert "<title>BTC Cell Explorer</title>" in response.text
    assert 'href="/brand/btc.ico"' in response.text


def test_served_webmanifest_carries_brand(tmp_path: Path, shell_dir: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"shortName": "BTC"}))

    app = create_app(Settings(static_dir=shell_dir, brand_dir=brand_dir))
    client = TestClient(app)

    assert json.loads(client.get("/site.webmanifest").text)["short_name"] == "BTC"


def test_templated_shell_is_not_cached(tmp_path: Path, shell_dir: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "BTC"}))

    app = create_app(Settings(static_dir=shell_dir, brand_dir=brand_dir))
    response = TestClient(app).get("/some/spa/route")
    assert response.headers["cache-control"] == "no-cache"


def test_unbranded_shell_is_byte_identical_to_the_file(shell_dir: Path):
    app = create_app(Settings(static_dir=shell_dir))
    response = TestClient(app).get("/some/spa/route")
    assert response.text == _read(shell_dir, "index.html")
