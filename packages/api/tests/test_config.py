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


def test_settings_keycloak_defaults():
    """Auth settings are all None by default (auth disabled)."""
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.keycloak_url is None
    assert settings.keycloak_realm is None
    assert settings.keycloak_client_id is None
    assert settings.keycloak_client_secret is None
    assert settings.cors_origins == ""
    assert settings.cors_origin_list == []


def test_settings_keycloak_from_env(monkeypatch: "pytest.MonkeyPatch"):
    """Auth settings are populated from environment variables."""
    monkeypatch.setenv("KEYCLOAK_URL", "https://auth.example.com")
    monkeypatch.setenv("KEYCLOAK_REALM", "cbioportal")
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "cell-explorer")
    monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "secret123")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,https://portal.example.com")
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.keycloak_url == "https://auth.example.com"
    assert settings.keycloak_realm == "cbioportal"
    assert settings.keycloak_client_id == "cell-explorer"
    assert settings.keycloak_client_secret == "secret123"
    assert settings.cors_origin_list == ["http://localhost:3000", "https://portal.example.com"]


def test_settings_auth_enabled_property(monkeypatch: "pytest.MonkeyPatch"):
    """auth_enabled is True only when all required Keycloak fields are set."""
    from cell_explorer_api.config import Settings

    assert Settings().auth_enabled is False

    monkeypatch.setenv("KEYCLOAK_URL", "https://auth.example.com")
    monkeypatch.setenv("KEYCLOAK_REALM", "cbioportal")
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "cell-explorer")
    monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "secret123")
    assert Settings().auth_enabled is True
