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
    # Bytes, not text: the shell is now decoded at startup and re-encoded on
    # the way out, and only a byte comparison pins that round-trip.
    assert response.content == (shell_dir / "index.html").read_bytes()


# --- Finding 2: the resolved-path comparison ------------------------------


def test_index_html_requested_directly_is_branded(tmp_path: Path, shell_dir: Path):
    """/index.html is a real file under the static root, so it would otherwise
    be served raw by the file branch and bypass templating entirely — silently,
    and only where STATIC_DIR resolves through a symlink."""
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps({"name": "BTC", "favicon": {"ico": "btc.ico"}})
    )
    (brand_dir / "btc.ico").write_bytes(b"\x00")

    client = TestClient(create_app(Settings(static_dir=shell_dir, brand_dir=brand_dir)))

    response = client.get("/index.html")
    assert "<title>BTC Cell Explorer</title>" in response.text
    assert 'href="/brand/btc.ico"' in response.text


# --- Finding 1: a malformed site.webmanifest must never raise --------------


def test_webmanifest_json_array_is_returned_unchanged():
    original = "[1,2]"
    assert render_webmanifest(original, _brand({"name": "BTC"})) == original


def test_webmanifest_json_string_is_returned_unchanged():
    original = '"hi"'
    assert render_webmanifest(original, _brand({"name": "BTC"})) == original


def test_webmanifest_invalid_json_is_returned_unchanged():
    original = "{not json"
    assert render_webmanifest(original, _brand({"name": "BTC"})) == original


def test_webmanifest_icons_not_a_list_is_left_alone():
    out = render_webmanifest('{"icons": {"a": "b"}}', _brand({"name": "BTC"}))
    parsed = json.loads(out)
    assert parsed["icons"] == {"a": "b"}  # untouched, not crashed
    assert parsed["short_name"] == "BTC"  # the rest is still branded


def test_webmanifest_icon_entries_that_are_not_objects_are_skipped():
    out = render_webmanifest('{"icons": ["x"]}', _brand({"name": "BTC"}))
    parsed = json.loads(out)
    assert parsed["icons"] == ["x"]
    assert parsed["short_name"] == "BTC"


def test_webmanifest_icon_with_unhashable_sizes_is_skipped():
    """`sizes` is looked up in a dict, so a non-string would raise TypeError."""
    out = render_webmanifest(
        '{"icons": [{"sizes": ["192x192"], "src": "./a.png"}]}',
        _brand({"name": "BTC", "favicon": {"png192": "btc-192.png"}}),
    )
    assert json.loads(out)["icons"][0]["src"] == "./a.png"


def test_malformed_webmanifest_does_not_stop_the_app_booting(
    tmp_path: Path, shell_dir: Path
):
    """The outage case: a non-default brand plus a hand-edited manifest."""
    (shell_dir / "site.webmanifest").write_text("[1,2]")
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "BTC"}))

    client = TestClient(create_app(Settings(static_dir=shell_dir, brand_dir=brand_dir)))

    assert client.get("/site.webmanifest").text == "[1,2]"
    assert "<title>BTC Cell Explorer</title>" in client.get("/some/spa/route").text


# --- Finding 5b: the served unbranded webmanifest --------------------------


def test_served_webmanifest_is_unchanged_for_default_brand(shell_dir: Path):
    client = TestClient(create_app(Settings(static_dir=shell_dir)))

    response = client.get("/site.webmanifest")
    assert response.content == (shell_dir / "site.webmanifest").read_bytes()
    assert response.headers["content-type"].startswith("application/manifest+json")
    assert response.headers["cache-control"] == "no-cache"


# --- Finding 3: the shell is decoded at startup now ------------------------


def test_non_ascii_shell_survives_the_decode_round_trip(
    tmp_path: Path, shell_dir: Path
):
    """The old FileResponse never decoded the file; this one does."""
    original = _read(shell_dir, "index.html").replace(
        "<title>cBioPortal Cell Explorer</title>",
        "<title>cBioPortal — Cell Explorer © 2026</title>",
    )
    (shell_dir / "index.html").write_text(original, encoding="utf-8")

    unbranded = TestClient(create_app(Settings(static_dir=shell_dir))).get("/route")
    assert unbranded.content == (shell_dir / "index.html").read_bytes()

    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps({"title": "Ünïcode — Explorer ©"}), encoding="utf-8"
    )
    branded = TestClient(
        create_app(Settings(static_dir=shell_dir, brand_dir=brand_dir))
    ).get("/route")
    assert "<title>Ünïcode — Explorer ©</title>" in branded.text


def test_undecodable_shell_degrades_to_the_previous_behavior(
    tmp_path: Path, shell_dir: Path
):
    """A shell that cannot be decoded must not stop the app from booting."""
    raw = b"<!doctype html><title>caf\xe9</title>"  # latin-1: invalid utf-8
    (shell_dir / "index.html").write_bytes(raw)
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "BTC"}))

    client = TestClient(create_app(Settings(static_dir=shell_dir, brand_dir=brand_dir)))

    response = client.get("/some/spa/route")
    assert response.status_code == 200
    assert response.content == raw
    # Falls all the way back to the old FileResponse, etag and all.
    assert "etag" in response.headers


# --- Fix 6: no-cache with a validator still 304s ---------------------------


def _shell_client(tmp_path: Path, shell_dir: Path) -> TestClient:
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "BTC"}))
    return TestClient(create_app(Settings(static_dir=shell_dir, brand_dir=brand_dir)))


def test_shell_carries_an_etag_and_revalidates_to_304(tmp_path: Path, shell_dir: Path):
    client = _shell_client(tmp_path, shell_dir)

    first = client.get("/some/spa/route")
    assert first.status_code == 200
    etag = first.headers["etag"]
    assert first.headers["cache-control"] == "no-cache"

    second = client.get("/some/spa/route", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == etag


def test_webmanifest_carries_an_etag_and_revalidates_to_304(
    tmp_path: Path, shell_dir: Path
):
    client = _shell_client(tmp_path, shell_dir)

    first = client.get("/site.webmanifest")
    etag = first.headers["etag"]

    second = client.get("/site.webmanifest", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.content == b""


def test_a_stale_etag_still_gets_the_body(tmp_path: Path, shell_dir: Path):
    client = _shell_client(tmp_path, shell_dir)

    response = client.get("/some/spa/route", headers={"If-None-Match": '"stale"'})
    assert response.status_code == 200
    assert "<title>BTC Cell Explorer</title>" in response.text


def test_the_shell_etag_tracks_the_brand(tmp_path: Path, shell_dir: Path):
    """Two deployments of the same shell under different brands must not share
    a validator — otherwise a rebrand serves a 304 for the old markup."""
    unbranded = TestClient(create_app(Settings(static_dir=shell_dir)))
    branded = _shell_client(tmp_path, shell_dir)

    assert (
        unbranded.get("/route").headers["etag"]
        != branded.get("/route").headers["etag"]
    )
