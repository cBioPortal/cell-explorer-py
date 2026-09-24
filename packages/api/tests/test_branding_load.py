import json
import logging
from pathlib import Path

from cell_explorer_api.branding import DEFAULT_BRAND, load_brand


def _bundle(tmp_path: Path, config: dict, assets: tuple[str, ...] = ()) -> Path:
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps(config))
    for asset in assets:
        (brand_dir / asset).write_bytes(b"<svg/>")
    return brand_dir


def test_none_brand_dir_yields_default():
    assert load_brand(None) == DEFAULT_BRAND


def test_missing_directory_yields_default(tmp_path: Path, caplog):
    with caplog.at_level(logging.WARNING):
        brand = load_brand(tmp_path / "nonexistent")
    assert brand == DEFAULT_BRAND
    assert "brand" in caplog.text.lower()


def test_missing_brand_json_yields_default(tmp_path: Path, caplog):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    with caplog.at_level(logging.WARNING):
        brand = load_brand(brand_dir)
    assert brand == DEFAULT_BRAND
    assert "brand.json" in caplog.text


def test_malformed_json_yields_default(tmp_path: Path, caplog):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text("{not json")
    with caplog.at_level(logging.WARNING):
        brand = load_brand(brand_dir)
    assert brand == DEFAULT_BRAND
    assert "brand.json" in caplog.text


def test_json_that_is_not_an_object_yields_default(tmp_path: Path, caplog):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text('["a", "list"]')
    with caplog.at_level(logging.WARNING):
        brand = load_brand(brand_dir)
    assert brand == DEFAULT_BRAND


def test_valid_bundle_loads(tmp_path: Path):
    brand_dir = _bundle(
        tmp_path,
        {
            "name": "Break Through Cancer",
            "logo": {"onDark": "logo-white.svg"},
            "colors": {"ink": "#240D00"},
        },
        assets=("logo-white.svg",),
    )
    brand = load_brand(brand_dir)
    assert brand.name == "Break Through Cancer"
    assert brand.logo.on_dark == "logo-white.svg"
    assert brand.colors.ink == "#240D00"


def test_referenced_asset_missing_from_disk_is_dropped(tmp_path: Path, caplog):
    brand_dir = _bundle(
        tmp_path,
        {"name": "BTC", "logo": {"onDark": "logo-white.svg"}},
        assets=(),  # file not written
    )
    with caplog.at_level(logging.WARNING):
        brand = load_brand(brand_dir)
    assert brand.logo.on_dark is None
    assert brand.name == "BTC"  # the rest of the brand survives
    assert "logo-white.svg" in caplog.text


def test_missing_asset_does_not_drop_present_ones(tmp_path: Path):
    brand_dir = _bundle(
        tmp_path,
        {"logo": {"onDark": "logo-white.svg", "onLight": "logo.svg"}},
        assets=("logo.svg",),
    )
    brand = load_brand(brand_dir)
    assert brand.logo.on_dark is None
    assert brand.logo.on_light == "logo.svg"


def test_asset_that_is_a_directory_is_dropped(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"logo": {"onDark": "logo.svg"}}))
    (brand_dir / "logo.svg").mkdir()
    brand = load_brand(brand_dir)
    assert brand.logo.on_dark is None


def test_favicon_assets_checked_for_existence(tmp_path: Path):
    brand_dir = _bundle(
        tmp_path,
        {"favicon": {"ico": "favicon.ico", "svg": "favicon.svg"}},
        assets=("favicon.ico",),
    )
    brand = load_brand(brand_dir)
    assert brand.favicon.ico == "favicon.ico"
    assert brand.favicon.svg is None


def test_non_utf8_brand_json_yields_default(tmp_path: Path, caplog):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_bytes(b"\xff\xfe\x00invalid")
    with caplog.at_level(logging.WARNING):
        brand = load_brand(brand_dir)
    assert brand == DEFAULT_BRAND
    assert "brand.json" in caplog.text


def test_unstattable_directory_yields_default(tmp_path: Path, monkeypatch, caplog):
    def _boom(self):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(Path, "is_dir", _boom)
    with caplog.at_level(logging.WARNING):
        brand = load_brand(tmp_path / "brand")
    assert brand == DEFAULT_BRAND
    assert "BRAND_DIR" in caplog.text
