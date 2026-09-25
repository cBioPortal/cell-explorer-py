import json
from pathlib import Path

from cell_explorer_api.branding import DEFAULT_BRAND
from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


def test_brand_dir_defaults_to_none():
    assert Settings().brand_dir is None


def test_app_state_has_default_brand_when_unset():
    app = create_app(Settings())
    assert app.state.brand == DEFAULT_BRAND


def test_app_state_carries_loaded_brand(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "Break Through Cancer"}))
    app = create_app(Settings(brand_dir=brand_dir))
    assert app.state.brand.name == "Break Through Cancer"


def test_app_starts_when_brand_dir_is_missing(tmp_path: Path):
    # The whole point of fail-soft: a bad BRAND_DIR must not stop the app.
    app = create_app(Settings(brand_dir=tmp_path / "nope"))
    assert app.state.brand == DEFAULT_BRAND
