"""Tests for auth route endpoints."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient

from cell_explorer_api.auth.oidc import OidcClient
from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


def _generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _make_settings() -> Settings:
    return Settings(
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
    )


def _make_token(private_key, **overrides) -> str:
    claims = {
        "sub": "user-123",
        "name": "Test User",
        "email": "test@example.com",
        "realm_access": {"roles": ["viewer"]},
        "iss": "https://auth.example.com/realms/test-realm",
        "aud": "test-client",
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
    } | overrides
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-kid"})


@pytest.fixture()
def rsa_keys():
    return _generate_rsa_keypair()


@pytest.fixture()
def auth_client(rsa_keys):
    _, public_key = rsa_keys
    settings = _make_settings()
    app = create_app(settings)
    oidc: OidcClient = app.state.oidc
    oidc._apply_discovery({
        "issuer": "https://auth.example.com/realms/test-realm",
        "authorization_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/auth",
        "token_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/token",
        "jwks_uri": "https://auth.example.com/realms/test-realm/protocol/openid-connect/certs",
        "end_session_endpoint": "https://auth.example.com/realms/test-realm/protocol/openid-connect/logout",
    })
    oidc._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    return TestClient(app)


def test_login_redirects_to_keycloak(auth_client):
    response = auth_client.get("/api/auth/login", follow_redirects=False)
    assert response.status_code == 307
    location = response.headers["location"]
    assert "auth.example.com/realms/test-realm/protocol/openid-connect/auth" in location
    assert "client_id=test-client" in location


def test_login_sets_state_cookie(auth_client):
    response = auth_client.get("/api/auth/login", follow_redirects=False)
    assert "cce_state" in response.cookies


def test_cookies_not_secure_on_http(auth_client):
    """Cookies should not have the secure flag when served over HTTP."""
    response = auth_client.get("/api/auth/login", follow_redirects=False)
    set_cookie = response.headers.get_list("set-cookie")
    state_cookie = [c for c in set_cookie if c.startswith("cce_state")][0]
    assert "Secure" not in state_cookie


def test_cookies_secure_with_forwarded_proto(auth_client):
    """Cookies should have the secure flag when x-forwarded-proto is https."""
    response = auth_client.get(
        "/api/auth/login",
        headers={"x-forwarded-proto": "https"},
        follow_redirects=False,
    )
    set_cookie = response.headers.get_list("set-cookie")
    state_cookie = [c for c in set_cookie if c.startswith("cce_state")][0]
    assert "Secure" in state_cookie


def test_me_returns_user_when_authenticated(auth_client, rsa_keys):
    private_key, _ = rsa_keys
    token = _make_token(private_key)
    auth_client.cookies.set("cce_access", token)
    response = auth_client.get("/api/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["sub"] == "user-123"
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"


def test_me_returns_401_when_not_authenticated(auth_client):
    response = auth_client.get("/api/auth/me")
    assert response.status_code == 401


def _mock_end_session(client, **kwargs):
    import unittest.mock as mock

    end_session = mock.AsyncMock(**kwargs)
    client.app.state.oidc.end_session = end_session
    return end_session


def _set_cookie_headers(response, name):
    return [c for c in response.headers.get_list("set-cookie") if c.startswith(f"{name}=")]


def test_logout_clears_cookies(auth_client, rsa_keys):
    _mock_end_session(auth_client)
    private_key, _ = rsa_keys
    token = _make_token(private_key)
    auth_client.cookies.set("cce_access", token)
    auth_client.cookies.set("cce_refresh", "some-refresh-token")
    response = auth_client.post("/api/auth/logout")
    assert response.status_code == 200
    for name in ("cce_access", "cce_refresh"):
        assert any("Max-Age=0" in c for c in _set_cookie_headers(response, name))


def test_logout_ends_the_provider_session(auth_client):
    end_session = _mock_end_session(auth_client)
    auth_client.cookies.set("cce_refresh", "some-refresh-token")
    auth_client.post("/api/auth/logout")
    end_session.assert_awaited_once_with("some-refresh-token")


def test_logout_without_a_refresh_cookie_skips_the_provider(auth_client):
    end_session = _mock_end_session(auth_client)
    response = auth_client.post("/api/auth/logout")
    assert response.status_code == 200
    end_session.assert_not_awaited()


def test_logout_still_signs_out_when_the_provider_fails(auth_client):
    _mock_end_session(auth_client, side_effect=Exception("unreachable"))
    auth_client.cookies.set("cce_refresh", "some-refresh-token")
    response = auth_client.post("/api/auth/logout")
    assert response.status_code == 200
    assert any("Max-Age=0" in c for c in _set_cookie_headers(response, "cce_refresh"))


def test_logout_marks_the_browser_as_signed_out(auth_client):
    _mock_end_session(auth_client)
    response = auth_client.post("/api/auth/logout")
    [marker] = _set_cookie_headers(response, "cce_signed_out")
    assert "Path=/api/auth" in marker
    assert "HttpOnly" in marker


def test_login_after_a_sign_out_asks_which_account(auth_client):
    auth_client.cookies.set("cce_signed_out", "1", path="/api/auth")
    response = auth_client.get("/api/auth/login", follow_redirects=False)
    assert "prompt=select_account" in response.headers["location"]


def test_login_without_a_sign_out_is_silent(auth_client):
    response = auth_client.get("/api/auth/login", follow_redirects=False)
    assert "prompt=" not in response.headers["location"]


def test_me_sets_refreshed_cookies(auth_client, rsa_keys):
    """When the access token is expired but refresh succeeds, /me should set new cookies."""
    private_key, _ = rsa_keys

    # Create an expired access token. Past the JWT decode leeway window
    # (30s) so the decoder genuinely rejects it instead of tolerating
    # clock skew.
    expired_token = _make_token(private_key, exp=int(time.time()) - 60)

    # Create a fresh token that the refresh flow will return
    fresh_token = _make_token(private_key)

    # Mock the refresh_token method on the oidc client
    oidc: OidcClient = auth_client.app.state.oidc
    import unittest.mock as mock
    oidc.refresh_token = mock.AsyncMock(return_value={
        "access_token": fresh_token,
        "refresh_token": "new-refresh-token-value",
    })

    auth_client.cookies.set("cce_access", expired_token)
    auth_client.cookies.set("cce_refresh", "old-refresh-token")

    response = auth_client.get("/api/auth/me")
    assert response.status_code == 200
    assert response.json()["sub"] == "user-123"

    # Verify refreshed cookies are set with the correct values
    set_cookie_headers = response.headers.get_list("set-cookie")
    cookie_map = {}
    for header in set_cookie_headers:
        name, _, rest = header.partition("=")
        value = rest.split(";")[0]
        cookie_map[name] = value
    assert "cce_access" in cookie_map, "Expected cce_access cookie to be set after refresh"
    assert cookie_map["cce_access"] == fresh_token, "cce_access should contain the refreshed token"
    assert "cce_refresh" in cookie_map, "Expected cce_refresh cookie to be set after refresh"
    assert cookie_map["cce_refresh"] == "new-refresh-token-value", "cce_refresh should contain the new refresh token"


def test_refresh_endpoint_rotates_cookies(auth_client, rsa_keys):
    """/api/auth/refresh exchanges the refresh cookie for a new pair.

    Used by the frontend's proactive-refresh timer to keep the access
    cookie fresh before its TTL expires, avoiding the rotation race that
    happens when many parallel requests all try to refresh after expiry.
    """
    private_key, _ = rsa_keys
    fresh_access = _make_token(private_key)

    oidc: OidcClient = auth_client.app.state.oidc
    import unittest.mock as mock
    oidc.refresh_token = mock.AsyncMock(return_value={
        "access_token": fresh_access,
        "refresh_token": "rotated-refresh",
    })

    auth_client.cookies.set("cce_refresh", "current-refresh")
    response = auth_client.post("/api/auth/refresh")
    assert response.status_code == 204

    set_cookie_headers = response.headers.get_list("set-cookie")
    cookie_map = {}
    for header in set_cookie_headers:
        name, _, rest = header.partition("=")
        cookie_map[name] = rest.split(";")[0]
    assert cookie_map.get("cce_access") == fresh_access
    assert cookie_map.get("cce_refresh") == "rotated-refresh"


def test_refresh_endpoint_401_when_no_refresh_cookie(auth_client):
    """Without cce_refresh, /refresh returns 401 — caller should re-login."""
    response = auth_client.post("/api/auth/refresh")
    assert response.status_code == 401


def test_refresh_endpoint_401_when_keycloak_rejects(auth_client):
    """When the OIDC provider rejects the refresh token, /refresh returns 401."""
    oidc: OidcClient = auth_client.app.state.oidc
    import unittest.mock as mock
    oidc.refresh_token = mock.AsyncMock(side_effect=Exception("invalid_grant"))

    auth_client.cookies.set("cce_refresh", "expired-refresh")
    response = auth_client.post("/api/auth/refresh")
    assert response.status_code == 401


def test_auth_routes_return_401_without_credentials():
    """When no credentials are presented, auth routes return 401 regardless of keycloak config."""
    settings = Settings()
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


@pytest.mark.parametrize(
    "next_path, expected",
    [
        ("/view?dataset=spectrum", "/view?dataset=spectrum"),
        ("/collections/htan-vumc-crc", "/collections/htan-vumc-crc"),
        ("/", "/"),
        (None, "/"),
        ("", "/"),
        ("https://evil.example.com/", "/"),
        ("//evil.example.com/", "/"),
        ("/\\evil.example.com/", "/"),
        ("javascript:alert(1)", "/"),
        ("/view\r\nSet-Cookie: x=1", "/"),
        ("/api/auth/login", "/"),
        ("/api", "/"),
    ],
)
def test_safe_next_path(next_path, expected):
    from cell_explorer_api.routes.auth import _safe_next_path

    assert _safe_next_path(next_path) == expected


def _login_then_callback(client, monkeypatch, **login_params):
    login = client.get("/api/auth/login", params=login_params, follow_redirects=False)
    state = login.cookies["cce_state"]

    async def fake_exchange_code(code, redirect_uri):
        return {"access_token": "access", "refresh_token": "refresh"}

    monkeypatch.setattr(client.app.state.oidc, "exchange_code", fake_exchange_code)
    return client.get(
        "/api/auth/callback",
        params={"code": "the-code", "state": state},
        follow_redirects=False,
    )


def test_callback_returns_to_the_page_login_started_from(auth_client, monkeypatch):
    response = _login_then_callback(
        auth_client, monkeypatch, next="/view?dataset=spectrum&gene=CD8A"
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/view?dataset=spectrum&gene=CD8A"


def test_callback_returns_to_index_without_next(auth_client, monkeypatch):
    response = _login_then_callback(auth_client, monkeypatch)
    assert response.headers["location"] == "/"


def test_callback_ignores_an_offsite_next(auth_client, monkeypatch):
    response = _login_then_callback(auth_client, monkeypatch, next="//evil.example.com/")
    assert response.headers["location"] == "/"


def test_callback_clears_the_next_cookie(auth_client, monkeypatch):
    response = _login_then_callback(auth_client, monkeypatch, next="/collections/crc-metastasis")
    set_cookie = response.headers.get_list("set-cookie")
    assert any(c.startswith("cce_next=") and "Max-Age=0" in c for c in set_cookie)


def test_callback_clears_the_signed_out_marker(auth_client, monkeypatch):
    auth_client.cookies.set("cce_signed_out", "1", path="/api/auth")
    response = _login_then_callback(auth_client, monkeypatch)
    [marker] = _set_cookie_headers(response, "cce_signed_out")
    assert "Max-Age=0" in marker
    assert "Path=/api/auth" in marker


# --- moving anonymous chat conversations into the account on sign-in ---

def _anon_db(app, anon_id):
    """In-memory DB holding one anonymous thread with a message and its feedback."""
    import asyncio

    from sqlmodel import SQLModel
    from sqlmodel.ext.asyncio.session import AsyncSession

    from cell_explorer_api.db import create_engine
    from cell_explorer_api.db.models import (
        ChatFeedback, ChatMessageRow, ChatThread, Dataset, Datasource, DatasourceType,
    )

    engine = create_engine("sqlite+aiosqlite://")

    async def _seed():
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with AsyncSession(engine) as s:
            src = Datasource(name="cdn", type=DatasourceType.S3_CLOUDFRONT, base_url="https://cdn.example.com")
            s.add(src)
            await s.flush()
            ds = Dataset(datasource_id=src.id, name="A", slug="a", path="a.zarr", is_public=True)
            s.add(ds)
            await s.flush()
            thread = ChatThread(user_sub=f"anon:{anon_id}", dataset_id=ds.id, title="t")
            s.add(thread)
            await s.flush()
            msg = ChatMessageRow(thread_id=thread.id, role="assistant", content="hi")
            s.add(msg)
            await s.flush()
            s.add(ChatFeedback(message_id=msg.id, user_sub=f"anon:{anon_id}", rating="up"))
            await s.commit()

    asyncio.run(_seed())
    app.state.db_engine = engine
    return engine


def _subs(engine):
    import asyncio

    from sqlmodel import select
    from sqlmodel.ext.asyncio.session import AsyncSession

    from cell_explorer_api.db.models import ChatFeedback, ChatThread

    async def _go():
        async with AsyncSession(engine) as s:
            threads = [t.user_sub for t in (await s.exec(select(ChatThread))).all()]
            feedback = [f.user_sub for f in (await s.exec(select(ChatFeedback))).all()]
            return threads, feedback

    return asyncio.run(_go())


def _sign_in_with(client, monkeypatch, access_token):
    login = client.get("/api/auth/login", follow_redirects=False)

    async def fake_exchange_code(code, redirect_uri):
        return {"access_token": access_token, "refresh_token": "refresh"}

    monkeypatch.setattr(client.app.state.oidc, "exchange_code", fake_exchange_code)
    return client.get(
        "/api/auth/callback",
        params={"code": "c", "state": login.cookies["cce_state"]},
        follow_redirects=False,
    )


def test_callback_moves_anonymous_conversations_to_the_account(auth_client, rsa_keys, monkeypatch):
    import uuid as _uuid

    anon = str(_uuid.uuid4())
    engine = _anon_db(auth_client.app, anon)
    auth_client.cookies.set("cce_anon", anon, path="/api")
    response = _sign_in_with(auth_client, monkeypatch, _make_token(rsa_keys[0], sub="user-123"))
    assert response.status_code == 302
    assert _subs(engine) == (["user-123"], ["user-123"])
    [cleared] = _set_cookie_headers(response, "cce_anon")
    assert "Max-Age=0" in cleared and "Path=/api" in cleared


def test_callback_without_anon_cookie_leaves_anonymous_conversations(auth_client, rsa_keys, monkeypatch):
    import uuid as _uuid

    anon = str(_uuid.uuid4())
    engine = _anon_db(auth_client.app, anon)
    response = _sign_in_with(auth_client, monkeypatch, _make_token(rsa_keys[0], sub="user-123"))
    assert response.status_code == 302
    assert _subs(engine) == ([f"anon:{anon}"], [f"anon:{anon}"])
    assert _set_cookie_headers(response, "cce_anon") == []


def test_callback_with_nothing_to_claim_still_signs_in(auth_client, rsa_keys, monkeypatch):
    import uuid as _uuid

    _anon_db(auth_client.app, str(_uuid.uuid4()))
    auth_client.cookies.set("cce_anon", str(_uuid.uuid4()), path="/api")
    response = _sign_in_with(auth_client, monkeypatch, _make_token(rsa_keys[0], sub="user-123"))
    assert response.status_code == 302
    assert any("cce_access=" in c for c in response.headers.get_list("set-cookie"))
