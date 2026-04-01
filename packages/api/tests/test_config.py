from pathlib import Path


def test_settings_defaults():
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.static_dir is None


def test_settings_static_dir_from_env(monkeypatch: "pytest.MonkeyPatch", static_dir: Path):
    monkeypatch.setenv("STATIC_DIR", str(static_dir))

    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.static_dir == static_dir


def test_settings_validates_static_dir_exists(tmp_path: Path):
    from cell_explorer_api.config import validate_static_dir

    # Non-existent path
    result = validate_static_dir(tmp_path / "nonexistent")
    assert result is None

    # Exists but no index.html
    empty = tmp_path / "empty"
    empty.mkdir()
    result = validate_static_dir(empty)
    assert result is None


def test_settings_validates_static_dir_with_index(static_dir: Path):
    from cell_explorer_api.config import validate_static_dir

    result = validate_static_dir(static_dir)
    assert result == static_dir
