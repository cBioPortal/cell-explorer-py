"""OAuth routes for MCP sign-in: SDK handlers, root metadata and the consent page."""

import base64
import hashlib
import re
import secrets
import time
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.oauth_models import OAuthCode, OAuthPendingAuthorization
from cell_explorer_api.main import create_app
from cell_explorer_api.routes.auth import _safe_next_path
from cell_explorer_api.services import oauth_store as store

BASE = "https://ce.test"
ISSUER = f"{BASE}/api/oauth"
RES = "https://mcp.test/mcp"
REDIRECT = "http://localhost:3000/callback"
KC_ISSUER = "https://auth.example.com/realms/test-realm"


def _settings(**overrides) -> Settings:
    values = dict(
        database_url="sqlite+aiosqlite://",
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
        public_url=BASE,
    )
    values.update(overrides)
    return Settings(**values)


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with AsyncSession(eng) as db:
            await store.create_mcp_resource(db, url=RES, name="Cell Explorer <MCP>")
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def app(engine, signing_key):
    app = create_app(_settings())
    oidc = app.state.oidc
    oidc._apply_discovery({
        "issuer": KC_ISSUER,
        "authorization_endpoint": f"{KC_ISSUER}/protocol/openid-connect/auth",
        "token_endpoint": f"{KC_ISSUER}/protocol/openid-connect/token",
        "jwks_uri": f"{KC_ISSUER}/protocol/openid-connect/certs",
        "end_session_endpoint": f"{KC_ISSUER}/protocol/openid-connect/logout",
    })
    oidc._jwks = {
        "test-kid": signing_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    app.state.db_engine = engine
    return app


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        yield c


def _session_cookie(signing_key, email="ada@example.org") -> dict:
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "user-1",
            "name": "Ada",
            "email": email,
            "realm_access": {"roles": ["lab"]},
            "iss": KC_ISSUER,
            "aud": "test-client",
            "exp": now + 300,
            "iat": now,
        },
        signing_key,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )
    return {"Cookie": f"cce_access={token}"}


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


async def _register(client, name="Claude <script>", redirect=REDIRECT) -> str:
    resp = await client.post(
        "/api/oauth/register",
        json={
            "redirect_uris": [redirect],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "client_name": name,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["client_id"]


async def _authorize(client, client_id: str, challenge: str, redirect=REDIRECT, issuer=ISSUER) -> str:
    """Start /authorize; returns the pending request id from the consent URL."""
    resp = await client.get(
        "/api/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "st-1",
            "resource": RES,
        },
    )
    assert resp.status_code == 302, resp.text
    location = resp.headers["location"]
    assert location.startswith(f"{issuer}/consent?request=")
    return parse_qs(urlparse(location).query)["request"][0]


async def _consent_csrf(client, request_id: str, cookie: dict) -> str:
    resp = await client.get("/api/oauth/consent", params={"request": request_id}, headers=cookie)
    assert resp.status_code == 200, resp.text
    match = re.search(r'name="csrf" value="([^"]+)"', resp.text)
    assert match
    return match.group(1)


async def _count(engine, model) -> int:
    async with AsyncSession(engine) as db:
        return len((await db.exec(select(model))).all())


# Metadata + mounting


async def test_metadata_served_at_root_and_under_api_oauth(client):
    for path in ("/api/oauth/.well-known/oauth-authorization-server", "/.well-known/oauth-authorization-server/api/oauth"):
        resp = await client.get(path)
        assert resp.status_code == 200, path
        meta = resp.json()
        assert meta["issuer"] == "https://ce.test/api/oauth"
        assert meta["authorization_endpoint"] == "https://ce.test/api/oauth/authorize"
        assert meta["token_endpoint"] == "https://ce.test/api/oauth/token"
        assert meta["registration_endpoint"] == "https://ce.test/api/oauth/register"
        assert meta["code_challenge_methods_supported"] == ["S256"]


async def test_root_metadata_not_shadowed_by_spa(engine, static_dir):
    app = create_app(_settings(static_dir=static_dir))
    app.state.db_engine = engine
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        resp = await c.get("/.well-known/oauth-authorization-server/api/oauth")
    assert resp.status_code == 200
    assert resp.json()["issuer"] == ISSUER


async def test_unknown_well_known_paths_are_404_not_spa(engine, static_dir):
    (static_dir / ".well-known").mkdir()
    (static_dir / ".well-known" / "security.txt").write_text("Contact: x")
    app = create_app(_settings(static_dir=static_dir))
    app.state.db_engine = engine
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        for path in (
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-authorization-server",
            "/.well-known/openid-configuration",
            "/.well-known/oauth-authorization-server/other",
        ):
            resp = await c.get(path)
            assert resp.status_code == 404, path
            assert not resp.headers["content-type"].startswith("text/html"), path
        assert (await c.get("/.well-known/oauth-authorization-server/api/oauth")).status_code == 200
        assert (await c.get("/.well-known/security.txt")).text == "Contact: x"  # real files still served
        assert (await c.get("/some/page")).status_code == 200  # SPA routes unchanged


@pytest.mark.parametrize("overrides", [{"public_url": None}, {"keycloak_client_secret": None}])
async def test_oauth_routes_absent_without_public_url(overrides):
    app = create_app(_settings(**overrides))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
            for path in (
                "/.well-known/oauth-authorization-server/api/oauth",
                "/api/oauth/.well-known/oauth-authorization-server",
                "/api/oauth/consent?request=x",
            ):
                assert (await c.get(path)).status_code == 404, path
            assert (await c.post("/api/oauth/register", json={})).status_code == 404
            assert (await c.post("/api/oauth/token-exchange", data={})).status_code == 404
    finally:
        await app.state.db_engine.dispose()


# _safe_next_path


def test_safe_next_path_allows_only_consent():
    assert _safe_next_path("/api/oauth/consent?request=x") == "/api/oauth/consent?request=x"
    assert _safe_next_path("/api/oauth/consent?request=Ab_9-z") == "/api/oauth/consent?request=Ab_9-z"
    for bad in (
        "/api/oauth/token",
        "/api/admin",
        "/api/oauth/consent",
        "/api/oauth/consent/../../admin",
        "/api/oauth/consent/../../admin?request=x",
        "/api/oauth/consent%3Frequest=x",
        "/api/oauth/consent%2F..%2F..%2Fadmin?request=x",
        "/api/oauth/consentx?request=x",
        "/api/oauth/consent?request=x&next=//evil.test",
        "/api/oauth/consent?request=x\\@evil.test",
    ):
        assert _safe_next_path(bad) == "/", bad


# Consent page


async def test_consent_redirects_to_login_without_session(client):
    client_id = await _register(client)
    request_id = await _authorize(client, client_id, _pkce()[1])
    resp = await client.get("/api/oauth/consent", params={"request": request_id})
    assert resp.status_code == 302
    assert resp.headers["location"].startswith("/api/auth/login?next=%2Fapi%2Foauth%2Fconsent%3Frequest%3D")
    assert resp.headers["location"].endswith(request_id)


async def test_consent_page_names_client_resource_and_user(client, signing_key):
    client_id = await _register(client, name="Claude <script>")
    request_id = await _authorize(client, client_id, _pkce()[1])
    resp = await client.get(
        "/api/oauth/consent", params={"request": request_id}, headers=_session_cookie(signing_key, "ada&co@example.org")
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
    assert resp.headers["referrer-policy"] == "no-referrer"
    body = resp.text
    assert "Claude &lt;script&gt;" in body and "<script>" not in body
    assert "Cell Explorer &lt;MCP&gt;" in body
    assert RES in body
    assert "ada&amp;co@example.org" in body
    assert f'name="request" value="{request_id}"' in body


async def test_consent_unknown_request_is_an_error_page(client, signing_key):
    resp = await client.get("/api/oauth/consent", params={"request": "nope"}, headers=_session_cookie(signing_key))
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("text/html")
    assert "location" not in resp.headers


async def test_consent_post_requires_matching_csrf(client, engine, signing_key):
    cookie = _session_cookie(signing_key)
    client_id = await _register(client)
    request_id = await _authorize(client, client_id, _pkce()[1])
    csrf = await _consent_csrf(client, request_id, cookie)

    for form in (
        {"request": request_id, "csrf": "wrong", "decision": "approve"},
        {"request": request_id, "decision": "approve"},
        {"request": request_id, "csrf": "wrong", "decision": "deny"},
    ):
        resp = await client.post("/api/oauth/consent", data=form, headers=cookie)
        assert resp.status_code == 400, form
        assert "location" not in resp.headers
    assert await _count(engine, OAuthCode) == 0

    # No session: never approves, even with the right csrf.
    resp = await client.post("/api/oauth/consent", data={"request": request_id, "csrf": csrf, "decision": "approve"})
    assert resp.status_code == 302
    assert resp.headers["location"].startswith("/api/auth/login?")
    assert await _count(engine, OAuthCode) == 0

    # A cross-origin POST is refused.
    resp = await client.post(
        "/api/oauth/consent",
        data={"request": request_id, "csrf": csrf, "decision": "approve"},
        headers={**cookie, "Origin": "https://evil.test"},
    )
    assert resp.status_code == 400
    assert await _count(engine, OAuthCode) == 0


async def test_consent_deny_redirects_with_access_denied(client, engine, signing_key):
    cookie = _session_cookie(signing_key)
    client_id = await _register(client)
    request_id = await _authorize(client, client_id, _pkce()[1])
    csrf = await _consent_csrf(client, request_id, cookie)

    resp = await client.post(
        "/api/oauth/consent",
        data={"request": request_id, "csrf": csrf, "decision": "deny"},
        headers={**cookie, "Origin": BASE},
    )
    assert resp.status_code == 302
    location = urlparse(resp.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == REDIRECT
    assert parse_qs(location.query) == {"error": ["access_denied"], "state": ["st-1"]}
    assert await _count(engine, OAuthCode) == 0
    assert await _count(engine, OAuthPendingAuthorization) == 0


# Registration limit


def test_registration_limit_defaults_to_200_per_hour():
    assert Settings.model_fields["oauth_registrations_per_hour"].default == 200


async def test_registration_is_rate_limited(engine):
    app = create_app(_settings(oauth_registrations_per_hour=3))
    app.state.db_engine = engine
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        for _ in range(3):
            await _register(c)
        resp = await c.post(
            "/api/oauth/register",
            json={"redirect_uris": [REDIRECT], "token_endpoint_auth_method": "none"},
        )
    assert resp.status_code == 429


# End to end


async def test_full_code_flow(client, signing_key):
    cookie = _session_cookie(signing_key)
    verifier, challenge = _pkce()
    client_id = await _register(client)
    request_id = await _authorize(client, client_id, challenge)
    csrf = await _consent_csrf(client, request_id, cookie)

    resp = await client.post(
        "/api/oauth/consent",
        data={"request": request_id, "csrf": csrf, "decision": "approve"},
        headers={**cookie, "Origin": BASE},
    )
    assert resp.status_code == 302
    location = urlparse(resp.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == REDIRECT
    query = parse_qs(location.query)
    assert query["state"] == ["st-1"]
    code = query["code"][0]

    token_form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT,
        "client_id": client_id,
        "resource": RES,
    }
    # The SDK checks PKCE: a wrong verifier is refused.
    bad = await client.post("/api/oauth/token", data={**token_form, "code_verifier": _pkce()[0]})
    assert bad.status_code == 400
    assert bad.json()["error"] == "invalid_grant"

    resp = await client.post("/api/oauth/token", data={**token_form, "code_verifier": verifier})
    assert resp.status_code == 200, resp.text
    tokens = resp.json()
    assert tokens["token_type"].lower() == "bearer"
    assert tokens["access_token"] and tokens["refresh_token"]

    resp = await client.post(
        "/api/oauth/token",
        data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"], "client_id": client_id},
    )
    assert resp.status_code == 200, resp.text
    refreshed = resp.json()
    assert refreshed["access_token"] != tokens["access_token"]
    assert refreshed["refresh_token"] != tokens["refresh_token"]


async def test_consent_page_shows_redirect_hostname_not_userinfo(client, signing_key):
    redirect = "https://claude.ai@evil.test/cb"
    client_id = await _register(client, redirect=redirect)
    request_id = await _authorize(client, client_id, _pkce()[1], redirect=redirect)
    resp = await client.get("/api/oauth/consent", params={"request": request_id}, headers=_session_cookie(signing_key))
    assert resp.status_code == 200
    assert "<code>https://evil.test</code>" in resp.text
    assert "claude.ai" not in resp.text


async def _approve(client, signing_key, headers: dict | None = None, issuer=ISSUER):
    cookie = _session_cookie(signing_key)
    client_id = await _register(client)
    request_id = await _authorize(client, client_id, _pkce()[1], issuer=issuer)
    csrf = await _consent_csrf(client, request_id, cookie)
    return await client.post(
        "/api/oauth/consent",
        data={"request": request_id, "csrf": csrf, "decision": "approve"},
        headers={**cookie, **(headers or {})},
    )


async def test_consent_approve_without_origin_header(client, signing_key):
    resp = await _approve(client, signing_key)
    assert resp.status_code == 302
    assert resp.headers["location"].startswith(REDIRECT + "?")
    assert "code" in parse_qs(urlparse(resp.headers["location"]).query)


async def test_consent_origin_compared_normalized(engine, signing_key):
    app = create_app(_settings(public_url="https://CE.test:443"))
    oidc = app.state.oidc
    oidc._apply_discovery({
        "issuer": KC_ISSUER,
        "authorization_endpoint": f"{KC_ISSUER}/protocol/openid-connect/auth",
        "token_endpoint": f"{KC_ISSUER}/protocol/openid-connect/token",
        "jwks_uri": f"{KC_ISSUER}/protocol/openid-connect/certs",
        "end_session_endpoint": f"{KC_ISSUER}/protocol/openid-connect/logout",
    })
    oidc._jwks = {
        "test-kid": signing_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    app.state.db_engine = engine
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        for origin in ("https://ce.test", "HTTPS://CE.TEST:443"):
            resp = await _approve(c, signing_key, {"Origin": origin}, issuer="https://CE.test:443/api/oauth")
            assert resp.status_code == 302, origin
            assert resp.headers["location"].startswith(REDIRECT + "?")
        resp = await _approve(c, signing_key, {"Origin": "https://ce.test:8443"}, issuer="https://CE.test:443/api/oauth")
        assert resp.status_code == 400


@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_consent_non_ascii_csrf_is_rejected(client, engine, signing_key, decision):
    cookie = _session_cookie(signing_key)
    client_id = await _register(client)
    request_id = await _authorize(client, client_id, _pkce()[1])
    resp = await client.post(
        "/api/oauth/consent",
        data={"request": request_id, "csrf": "caf\u00e9", "decision": decision},
        headers=cookie,
    )
    assert resp.status_code == 400
    assert "location" not in resp.headers
    assert await _count(engine, OAuthCode) == 0


# Loopback redirect URIs (RFC 8252 7.3)


def _authorize_params(client_id: str, challenge: str, redirect: str) -> dict:
    return {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "st-1",
        "resource": RES,
    }


async def test_loopback_redirect_matches_any_port(client, signing_key):
    cookie = _session_cookie(signing_key)
    verifier, challenge = _pkce()
    client_id = await _register(client, redirect="http://127.0.0.1:1234/callback")
    redirect = "http://127.0.0.1:5678/callback"
    request_id = await _authorize(client, client_id, challenge, redirect=redirect)
    csrf = await _consent_csrf(client, request_id, cookie)
    resp = await client.post(
        "/api/oauth/consent",
        data={"request": request_id, "csrf": csrf, "decision": "approve"},
        headers={**cookie, "Origin": BASE},
    )
    assert resp.status_code == 302
    assert resp.headers["location"].startswith(redirect + "?")
    code = parse_qs(urlparse(resp.headers["location"]).query)["code"][0]

    token_form = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "code_verifier": verifier,
        "resource": RES,
    }
    # The code is bound to the port it was requested with.
    other_port = await client.post("/api/oauth/token", data={**token_form, "redirect_uri": "http://127.0.0.1:1234/callback"})
    assert other_port.status_code == 400
    resp = await client.post("/api/oauth/token", data={**token_form, "redirect_uri": redirect})
    assert resp.status_code == 200, resp.text
    assert resp.json()["access_token"]


@pytest.mark.parametrize(
    ("registered", "requested"),
    [
        ("http://127.0.0.1:1234/callback", "http://127.0.0.1:5678/other"),  # path
        ("http://127.0.0.1:1234/callback", "http://localhost:5678/callback"),  # host spelled differently
        ("http://localhost:1234/callback", "http://127.0.0.1:5678/callback"),
        ("http://127.0.0.1:1234/callback", "http://127.0.0.1:5678/callback?x=1"),  # query
        ("http://127.0.0.1:1234/callback", "http://u@127.0.0.1:5678/callback"),  # userinfo
        ("https://app.test:8443/callback", "https://app.test:9443/callback"),  # not loopback
        ("http://app.test:1234/callback", "http://app.test:5678/callback"),
    ],
)
async def test_loopback_port_relaxation_is_narrow(client, registered, requested):
    client_id = await _register(client, redirect=registered)
    resp = await client.get("/api/oauth/authorize", params=_authorize_params(client_id, _pkce()[1], requested))
    assert resp.status_code == 400, requested
    assert "location" not in resp.headers


async def test_authorize_refused_beyond_pending_cap(client):
    client_id = await _register(client)
    for _ in range(50):
        await _authorize(client, client_id, _pkce()[1])
    resp = await client.get("/api/oauth/authorize", params=_authorize_params(client_id, _pkce()[1], REDIRECT))
    assert resp.status_code == 302
    location = urlparse(resp.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == REDIRECT
    assert parse_qs(location.query)["error"] == ["temporarily_unavailable"]
