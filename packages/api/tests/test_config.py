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


def test_cookie_max_age_defaults():
    """Cookie lifetimes default to 5 min access, 24h refresh."""
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.access_cookie_max_age == 300
    assert settings.refresh_cookie_max_age == 86400


def test_cookie_max_age_from_env(monkeypatch: "pytest.MonkeyPatch"):
    monkeypatch.setenv("ACCESS_COOKIE_MAX_AGE", "120")
    monkeypatch.setenv("REFRESH_COOKIE_MAX_AGE", "604800")
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.access_cookie_max_age == 120
    assert settings.refresh_cookie_max_age == 604800


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


def test_database_url_default():
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.database_url is None
    assert "cell_explorer.db" in settings.effective_database_url


def test_database_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./custom.db")
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.database_url == "sqlite+aiosqlite:///./custom.db"


def test_admin_api_key_default_none():
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.admin_api_key is None


def test_admin_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-key-123")
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.admin_api_key == "test-key-123"


def test_admin_enabled_when_key_set(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-key-123")
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.admin_enabled is True


def test_admin_disabled_when_no_key():
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.admin_enabled is False


def test_chat_enabled_when_key_set():
    from cell_explorer_api.config import Settings

    settings = Settings(anthropic_api_key="sk-ant-test")
    assert settings.chat_enabled is True


def test_chat_disabled_when_no_key():
    from cell_explorer_api.config import Settings

    settings = Settings()
    assert settings.chat_enabled is False


def test_keycloak_backward_compat_resolution():
    # Only KEYCLOAK_* set (existing prod shape) — provider defaults to keycloak.
    from cell_explorer_api.config import Settings

    s = Settings(
        keycloak_url="https://auth.example.com",
        keycloak_realm="cell-explorer",
        keycloak_client_id="ce-app",
        keycloak_client_secret="secret",
    )
    assert s.auth_provider == "keycloak"
    assert s.auth_enabled is True
    assert s.resolved_issuer == "https://auth.example.com/realms/cell-explorer"
    assert s.resolved_client_id == "ce-app"
    assert s.resolved_client_secret == "secret"
    assert s.resolved_audience == "ce-app"
    assert s.resolved_scopes == "openid profile email"
    assert s.resolved_roles_claims == [
        "realm_access.roles",
        "resource_access.ce-app.roles",
    ]
    assert s.resolved_idp_hint is None
    assert s.discovery_url == (
        "https://auth.example.com/realms/cell-explorer/.well-known/openid-configuration"
    )


def test_keycloak_idp_hint_flows_through():
    from cell_explorer_api.config import Settings

    s = Settings(
        keycloak_url="https://a", keycloak_realm="r",
        keycloak_client_id="c", keycloak_client_secret="x",
        keycloak_idp_hint="pingId",
    )
    assert s.resolved_idp_hint == "pingId"


def test_entra_resolution():
    from cell_explorer_api.config import Settings

    s = Settings(
        auth_provider="entra",
        oidc_issuer="https://login.microsoftonline.com/TENANT/v2.0",
        oidc_client_id="app-guid",
        oidc_client_secret="secret",
    )
    assert s.auth_enabled is True
    assert s.resolved_issuer == "https://login.microsoftonline.com/TENANT/v2.0"
    assert s.resolved_roles_claims == ["roles"]
    assert s.resolved_audience == "app-guid"
    assert "offline_access" in s.resolved_scopes.split()
    assert s.resolved_idp_hint is None


def test_roles_claims_override():
    from cell_explorer_api.config import Settings

    s = Settings(
        auth_provider="oidc",
        oidc_issuer="https://idp.example.com",
        oidc_client_id="c", oidc_client_secret="x",
        oidc_roles_claims="groups, my.custom.path",
    )
    assert s.resolved_roles_claims == ["groups", "my.custom.path"]


def test_oidc_client_creds_fall_back_to_keycloak_vars():
    from cell_explorer_api.config import Settings

    s = Settings(
        keycloak_url="https://a", keycloak_realm="r",
        keycloak_client_id="kc", keycloak_client_secret="kx",
    )
    assert s.resolved_client_id == "kc"
    assert s.resolved_client_secret == "kx"


def test_auth_disabled_when_incomplete():
    from cell_explorer_api.config import Settings

    assert Settings().auth_enabled is False
    assert Settings(auth_provider="entra", oidc_issuer="https://i").auth_enabled is False
