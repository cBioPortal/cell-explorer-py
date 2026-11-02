"""Ambient CloudFront signed cookies from the dataset access endpoint.

The browser cannot attach the signed cookies itself: zarrstore fires hundreds of
chunk requests straight at CloudFront, so the cookies must already be in the
cookie jar for the CDN's host. The access endpoint therefore sets them with
Set-Cookie, scoped to the datasource's configured parent domain and to the
dataset's own path, so that two private datasets on the same distribution never
overwrite each other's cookies.
"""

import base64
import json
import time
from http.cookies import SimpleCookie

import jwt
import pytest_asyncio
from cell_explorer_api.auth.oidc import OidcClient
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

CF_COOKIES = ("CloudFront-Policy", "CloudFront-Signature", "CloudFront-Key-Pair-Id")
DATASET_PATH = "cell-explorer/team-a/study/store.zarr"


@pytest_asyncio.fixture()
async def cf_app(monkeypatch):
    """App with auth and a private dataset on an s3_cloudfront datasource."""
    cf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cf_pem = cf_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    monkeypatch.setenv("DATASOURCE_TEST_CF_KEY_PAIR_ID", "KTESTPAIR")
    monkeypatch.setenv("DATASOURCE_TEST_CF_PRIVATE_KEY", cf_pem)
    monkeypatch.setenv("DATASOURCE_TEST_CF_COOKIE_DOMAIN", ".example.org")

    jwt_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    settings = Settings(
        database_url="sqlite+aiosqlite://",
        keycloak_url="https://auth.example.com",
        keycloak_realm="test-realm",
        keycloak_client_id="test-client",
        keycloak_client_secret="test-secret",
    )
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
        "test-kid": jwt_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="Test CloudFront",
            type=DatasourceType.S3_CLOUDFRONT,
            base_url="https://assets.example.org",
            credential_ref="TEST_CF",
        )
        session.add(ds)
        await session.flush()
        session.add(
            Dataset(
                datasource_id=ds.id,
                name="Private Study",
                slug="private-study",
                path=DATASET_PATH,
                is_public=False,
                required_roles=["team-a"],
            )
        )
        await session.commit()

    app.state.db_engine = engine
    app.state.test_jwt_key = jwt_key
    return app


def _client(app, roles=None):
    client = TestClient(app)
    if roles is not None:
        token = jwt.encode(
            {
                "sub": "user-123",
                "name": "Test User",
                "realm_access": {"roles": roles},
                "iss": "https://auth.example.com/realms/test-realm",
                "aud": "test-client",
                "exp": int(time.time()) + 300,
                "iat": int(time.time()),
            },
            app.state.test_jwt_key,
            algorithm="RS256",
            headers={"kid": "test-kid"},
        )
        client.cookies.set("cce_access", token)
    return client


def _cf_set_cookies(response) -> dict[str, SimpleCookie]:
    """CloudFront-* Set-Cookie headers on the response, keyed by cookie name."""
    found = {}
    for header in response.headers.get_list("set-cookie"):
        parsed = SimpleCookie()
        parsed.load(header)
        for name, morsel in parsed.items():
            if name in CF_COOKIES:
                found[name] = morsel
    return found


def _granted_cookies(app) -> dict[str, SimpleCookie]:
    """Call /access as a user holding the role; require all three cookies."""
    response = _client(app, roles=["team-a"]).post("/api/datasets/private-study/access")
    assert response.status_code == 200
    cookies = _cf_set_cookies(response)
    assert set(cookies) == set(CF_COOKIES)
    return cookies


def _cf_b64_decode(value: str) -> bytes:
    return base64.b64decode(value.replace("-", "+").replace("_", "=").replace("~", "/"))


def test_access_sets_all_three_cloudfront_cookies(cf_app):
    _granted_cookies(cf_app)


def test_cookies_are_scoped_to_domain_and_dataset_path(cf_app):
    for name, morsel in _granted_cookies(cf_app).items():
        assert morsel["domain"] == ".example.org", name
        # Trailing slash: a sibling store whose name merely starts with this
        # one's (store.zarr-v2) must not receive these cookies.
        assert morsel["path"] == f"/{DATASET_PATH}/", name


def test_cookies_are_secure_httponly_lax(cf_app):
    for name, morsel in _granted_cookies(cf_app).items():
        assert morsel["secure"], name
        assert morsel["httponly"], name
        assert morsel["samesite"].lower() == "lax", name


def test_cookie_max_age_matches_credential_expiry(cf_app):
    max_age = int(_granted_cookies(cf_app)["CloudFront-Policy"]["max-age"])
    # DEFAULT_TTL_SECONDS is 1800; allow a little slack for test runtime.
    assert 1790 <= max_age <= 1800


def test_cookie_policy_grants_only_the_dataset_prefix(cf_app):
    policy = json.loads(_cf_b64_decode(_granted_cookies(cf_app)["CloudFront-Policy"].value))
    resource = policy["Statement"][0]["Resource"]
    assert resource == f"https://assets.example.org/{DATASET_PATH}/*"


def test_body_omits_cookie_values_once_they_are_set_as_cookies(cf_app):
    # HttpOnly is pointless if the same values sit in a JSON body script can read.
    response = _client(cf_app, roles=["team-a"]).post("/api/datasets/private-study/access")
    data = response.json()
    assert data["credential_type"] == "signed_cookies"
    assert data["url"] == f"https://assets.example.org/{DATASET_PATH}"
    assert "expires_at" in data
    assert "cookies" not in data


def test_without_cookie_domain_falls_back_to_body_only(cf_app, monkeypatch):
    # A deployment that has not configured the domain keeps today's behaviour
    # rather than setting a host-only cookie the CDN host would never see.
    monkeypatch.delenv("DATASOURCE_TEST_CF_COOKIE_DOMAIN")
    response = _client(cf_app, roles=["team-a"]).post("/api/datasets/private-study/access")
    assert response.status_code == 200
    assert _cf_set_cookies(response) == {}
    assert set(response.json()["cookies"]) == set(CF_COOKIES)


def test_no_cookies_without_required_role(cf_app):
    response = _client(cf_app, roles=["team-b"]).post("/api/datasets/private-study/access")
    assert response.status_code == 403
    assert _cf_set_cookies(response) == {}


def test_no_cookies_when_anonymous(cf_app):
    response = _client(cf_app).post("/api/datasets/private-study/access")
    assert response.status_code == 401
    assert _cf_set_cookies(response) == {}
