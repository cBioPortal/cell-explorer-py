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
    # Inject a fake OidcClient whose authorization_url we can assert on.
    fake_kc = MagicMock()
    fake_kc.authorization_url = MagicMock(return_value="http://kc.test/auth?...signed_state")
    fake_kc.fetch_jwks = AsyncMock()
    app.state.oidc = fake_kc
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
    app.state.oidc = fake_kc
    client = TestClient(app)
    resp = client.get(
        "/api/auth/cli-login",
        params={"redirect_uri": "http://localhost:53412/callback"},
        follow_redirects=False,
    )
    assert resp.status_code == 501
    assert "CLI_STATE_SECRET" in resp.text


def test_callback_cli_flow_returns_html_with_tokens():
    client, fake_kc, settings = _make_client()
    fake_kc.exchange_code = AsyncMock(return_value={
        "access_token": "ACC123",
        "refresh_token": "REF456",
        "expires_in": 300,
    })

    # Mint a valid state as the /cli-login endpoint would
    state = _sign_state_for_test(settings.cli_state_secret, "http://localhost:53412/callback")

    resp = client.get(
        "/api/auth/callback",
        params={"code": "code123", "state": state},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # Tokens are embedded in the page for the JS fetch
    assert "ACC123" in resp.text
    assert "REF456" in resp.text
    assert "http://localhost:53412/callback" in resp.text
    fake_kc.exchange_code.assert_awaited_once()


def test_callback_expired_cli_state_rejected():
    client, fake_kc, settings = _make_client()
    expired = _sign_state_for_test(
        settings.cli_state_secret, "http://localhost:53412/callback", exp_offset=-60
    )
    resp = client.get(
        "/api/auth/callback",
        params={"code": "c", "state": expired},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "expired" in resp.text.lower() or "invalid" in resp.text.lower()
    fake_kc.exchange_code.assert_not_called()


def _sign_state_for_test(secret: str, redirect_uri: str, *, exp_offset: int = 600) -> str:
    import time as _time
    now = int(_time.time())
    return jwt.encode(
        {
            "cli_flow": True,
            "redirect_uri": redirect_uri,
            "nonce": "test-nonce",
            "iat": now,
            "exp": now + exp_offset,
        },
        secret,
        algorithm="HS256",
    )
