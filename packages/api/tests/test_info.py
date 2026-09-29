import json
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient

from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


def test_info_returns_expected_fields(client: TestClient):
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == version("cell-explorer-api")
    assert data["environment"] == "development"
    # git_sha is autodetected when running in a git repo
    assert isinstance(data["git_sha"], str | None)


def test_info_reflects_settings():
    settings = Settings(environment="staging", git_sha="abc1234")
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["environment"] == "staging"
    assert data["git_sha"] == "abc1234"


def test_git_sha_autodetected_when_not_set():
    settings = Settings()
    # In a git repo, git_sha should be autodetected
    assert settings.git_sha is not None
    assert len(settings.git_sha) == 7


def test_info_includes_auth_enabled():
    settings = Settings(
        keycloak_url="https://auth.example.com",
        keycloak_realm="test",
        keycloak_client_id="client",
        keycloak_client_secret="secret",
    )
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/info")
    assert response.status_code == 200
    assert response.json()["auth_enabled"] is True


def test_info_auth_disabled_by_default(client: TestClient):
    response = client.get("/api/info")
    assert response.json()["auth_enabled"] is False


def test_info_includes_chat_enabled_true():
    settings = Settings(anthropic_api_key="sk-ant-test")
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/info")
    assert response.status_code == 200
    assert response.json()["chat_enabled"] is True


def test_info_chat_disabled_by_default(client: TestClient):
    response = client.get("/api/info")
    assert response.json()["chat_enabled"] is False


def test_info_includes_google_analytics_id_when_configured():
    settings = Settings(google_analytics_id="G-ABC1234567")
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/info")
    assert response.status_code == 200
    assert response.json()["google_analytics_id"] == "G-ABC1234567"


def test_info_google_analytics_id_is_null_by_default(client: TestClient):
    # Unset means untracked. The frontend loads no script when this is null,
    # which is also how the GitHub Pages build ends up untracked — it has no
    # backend to ask.
    assert client.get("/api/info").json()["google_analytics_id"] is None


def test_info_brand_is_null_when_unbranded():
    client = TestClient(create_app(Settings()))
    assert client.get("/api/info").json()["brand"] is None


def test_info_brand_is_populated_when_branded(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps(
            {
                "name": "Break Through Cancer",
                "shortName": "BTC",
                "logoHref": "https://breakthroughcancer.org",
                "logo": {"onDark": "logo-white.svg"},
                "colors": {"ink": "#240D00"},
            }
        )
    )
    (brand_dir / "logo-white.svg").write_text("<svg/>")

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))
    brand = client.get("/api/info").json()["brand"]

    assert brand["name"] == "Break Through Cancer"
    assert brand["short_name"] == "BTC"
    assert brand["logo_href"] == "https://breakthroughcancer.org"
    assert brand["colors"]["ink"] == "#240D00"


def test_info_resolves_logo_filenames_to_urls(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps({"logo": {"onDark": "logo-white.svg"}})
    )
    (brand_dir / "logo-white.svg").write_text("<svg/>")

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))
    brand = client.get("/api/info").json()["brand"]

    assert brand["logo_on_dark"] == "/brand/logo-white.svg"
    assert brand["logo_on_light"] is None


def test_info_brand_colors_include_derived_theme_color(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"colors": {"ink": "#240D00"}}))

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))
    colors = client.get("/api/info").json()["brand"]["colors"]

    assert colors["ink"] == "#240D00"
    assert colors["theme_color"] == "#240D00"


def test_info_logo_alt_uses_explicit_value(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(
        json.dumps({"name": "Break Through Cancer", "logo": {"alt": "BTC logo"}})
    )

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))
    brand = client.get("/api/info").json()["brand"]

    assert brand["logo_alt"] == "BTC logo"


def test_info_logo_alt_falls_back_to_name(tmp_path: Path):
    brand_dir = tmp_path / "brand"
    brand_dir.mkdir()
    (brand_dir / "brand.json").write_text(json.dumps({"name": "Break Through Cancer"}))

    client = TestClient(create_app(Settings(brand_dir=brand_dir)))
    brand = client.get("/api/info").json()["brand"]

    assert brand["logo_alt"] == "Break Through Cancer"
