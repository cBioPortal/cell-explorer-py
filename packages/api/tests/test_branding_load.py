import json
import logging
from pathlib import Path

import pytest

from cell_explorer_api.branding import (
    DEFAULT_BRAND,
    _existing_asset,
    load_brand,
    load_bundle,
)


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


# --- The bundle: one stat, and the exact set of servable filenames ---------


def test_bundle_directory_is_the_dir_when_usable(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "BTC"}))

    assert load_bundle(brand_dir).directory == brand_dir


@pytest.mark.parametrize("case", ["unset", "missing", "not_a_dir"])
def test_bundle_directory_is_none_when_unusable(tmp_path: Path, case: str):
    """`directory is None` is the single signal callers mount off."""
    if case == "unset":
        brand_dir = None
    elif case == "missing":
        brand_dir = tmp_path / "nope"
    else:
        brand_dir = tmp_path / "file"
        brand_dir.write_text("not a directory")

    bundle = load_bundle(brand_dir)
    assert bundle.directory is None
    assert bundle.brand == DEFAULT_BRAND


def test_bundle_directory_is_none_when_brand_dir_cannot_be_stat_d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """EACCES is the one errno Path.is_dir() re-raises instead of swallowing."""
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    real_is_dir = Path.is_dir

    def is_dir(self: Path, *args, **kwargs):
        if self == brand_dir:
            raise PermissionError(13, "Permission denied")
        return real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", is_dir)

    bundle = load_bundle(brand_dir)
    assert bundle.directory is None
    assert bundle.brand == DEFAULT_BRAND


def test_bundle_asset_filenames_are_exactly_the_resolved_ones(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps(
            {
                "name": "BTC",
                "logo": {"onDark": "logo.svg", "onLight": "gone.svg"},
                "favicon": {"ico": "btc.ico", "svg": "../escape.svg"},
            }
        )
    )
    (brand_dir / "logo.svg").write_text("<svg/>")
    (brand_dir / "btc.ico").write_bytes(b"\x00")
    (brand_dir / "unreferenced.png").write_bytes(b"\x00")

    # `gone.svg` does not exist, `../escape.svg` fails validation, and
    # `unreferenced.png` is nobody's asset.
    assert load_bundle(brand_dir).asset_filenames == frozenset(
        {"logo.svg", "btc.ico"}
    )


# --- An over-long filename must not raise out of the load -----------------


def test_overlong_asset_filename_degrades_the_field_not_the_load(
    tmp_path: Path, caplog
):
    """A stat the OS refuses to answer (ENAMETOOLONG) used to escape load_bundle
    and take create_app with it. The field drops; the rest of the brand stands."""
    brand_dir = _bundle(
        tmp_path,
        {
            "name": "BTC",
            "logo": {"onDark": "a" * 5000 + ".svg", "onLight": "ok.svg"},
        },
        assets=("ok.svg",),
    )

    with caplog.at_level(logging.WARNING):
        bundle = load_bundle(brand_dir)

    assert bundle.brand.name == "BTC"
    assert bundle.brand.logo.on_dark is None
    assert bundle.brand.logo.on_light == "ok.svg"
    assert bundle.asset_filenames == frozenset({"ok.svg"})
    assert any("onDark" in r.getMessage() for r in caplog.records)


def test_existing_asset_guard_survives_a_stat_that_cannot_be_answered(
    tmp_path: Path, caplog
):
    """The second layer, exercised directly: _asset's cap is what stops the
    known trigger, so this is the only way to reach the guard behind it."""
    with caplog.at_level(logging.WARNING):
        assert _existing_asset(tmp_path, "a" * 5000 + ".svg") is None
    assert any("Could not stat brand asset" in r.getMessage() for r in caplog.records)
