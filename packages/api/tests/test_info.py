from importlib.metadata import version

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
