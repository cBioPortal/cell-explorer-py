from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from fastapi.testclient import TestClient

from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


def _settings_with_keycloak() -> Settings:
    return Settings(
        keycloak_url="http://kc.test",
        keycloak_realm="test",
        keycloak_client_id="cli-test",
        keycloak_client_secret="secret",
        cli_state_secret="test-state-secret-1234567890",
    )


def _make_client() -> tuple[TestClient, MagicMock, Settings]:
    settings = _settings_with_keycloak()
    app = create_app(settings=settings)
    # Inject a fake KeycloakClient whose authorization_url we can assert on.
    fake_kc = MagicMock()
    fake_kc.authorization_url = MagicMock(return_value="http://kc.test/auth?...signed_state")
    fake_kc.fetch_jwks = AsyncMock()
    app.state.keycloak = fake_kc
    return TestClient(app), fake_kc, settings


def test_cli_login_rejects_non_localhost_redirect_uri():
    client, _, _ = _make_client()
    resp = client.get(
        "/api/auth/cli-login",
        params={"redirect_uri": "https://evil.example.com/callback"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "localhost" in resp.text.lower()


def test_cli_login_rejects_https_localhost():
    client, _, _ = _make_client()
    resp = client.get(
        "/api/auth/cli-login",
        params={"redirect_uri": "https://localhost:1234/callback"},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_cli_login_accepts_localhost_and_signs_state():
    client, fake_kc, settings = _make_client()
    resp = client.get(
        "/api/auth/cli-login",
        params={"redirect_uri": "http://localhost:53412/callback"},
        follow_redirects=False,
    )
    assert resp.status_code == 307
    assert resp.headers["location"].startswith("http://kc.test/auth")

    # The signed state was passed to authorization_url as its `state` arg
    call_kwargs = fake_kc.authorization_url.call_args.kwargs
    state = call_kwargs["state"]
    decoded = jwt.decode(state, settings.cli_state_secret, algorithms=["HS256"])
    assert decoded["cli_flow"] is True
    assert decoded["redirect_uri"] == "http://localhost:53412/callback"
    assert "nonce" in decoded
    assert "exp" in decoded


def test_cli_login_accepts_127_0_0_1():
    client, _, _ = _make_client()
    resp = client.get(
        "/api/auth/cli-login",
        params={"redirect_uri": "http://127.0.0.1:8888/callback"},
        follow_redirects=False,
    )
    assert resp.status_code == 307


def test_cli_login_requires_auth_enabled():
    # When keycloak_url is unset, auth is disabled → endpoint 501s.
    settings = Settings()
    app = create_app(settings=settings)
    client = TestClient(app)
    resp = client.get(
        "/api/auth/cli-login",
        params={"redirect_uri": "http://localhost:53412/callback"},
        follow_redirects=False,
    )
    assert resp.status_code == 501


def test_cli_login_requires_cli_state_secret():
    # Keycloak configured but cli_state_secret missing → 501.
    settings = Settings(
        keycloak_url="http://kc.test",
        keycloak_realm="test",
        keycloak_client_id="cli-test",
        keycloak_client_secret="secret",
        # cli_state_secret intentionally not set
    )
    app = create_app(settings=settings)
    fake_kc = MagicMock()
    fake_kc.fetch_jwks = AsyncMock()
    app.state.keycloak = fake_kc
    client = TestClient(app)
    resp = client.get(
        "/api/auth/cli-login",
        params={"redirect_uri": "http://localhost:53412/callback"},
        follow_redirects=False,
    )
    assert resp.status_code == 501
    assert "CLI_STATE_SECRET" in resp.text
