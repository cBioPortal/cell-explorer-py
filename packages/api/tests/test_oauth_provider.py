"""OAuth authorization-server provider for MCP sign-in, driven directly."""

import time
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import pytest_asyncio
from mcp.server.auth.provider import AuthorizationParams, AuthorizeError, TokenError
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.oauth_provider import CellExplorerOAuthProvider
from cell_explorer_api.db.models import utcnow
from cell_explorer_api.db.oauth_models import (
    OAuthCode,
    OAuthGrant,
    OAuthPendingAuthorization,
    OAuthToken,
)
from cell_explorer_api.services import oauth_store as store

ISSUER = "https://ce.example.org/api/oauth"
RES = "https://mcp.example.org/mcp"
REDIRECT = "http://localhost:3000/callback"
USER = User(sub="u1", name="A", email="a@b.c", roles=["r1", "r2"])


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def db(engine):
    async with AsyncSession(engine) as s:
        yield s


@pytest_asyncio.fixture
async def provider(engine, db):
    await store.create_mcp_resource(db, url=RES, name="MCP")
    p = CellExplorerOAuthProvider(session_factory=lambda: AsyncSession(engine), issuer_url=ISSUER)
    await p.register_client(CLIENT)
    return p


CLIENT = OAuthClientInformationFull(
    client_id="cid",
    redirect_uris=[AnyUrl(REDIRECT)],
    token_endpoint_auth_method="none",
    grant_types=["authorization_code", "refresh_token"],
)


def _params(resource: str | None = RES) -> AuthorizationParams:
    return AuthorizationParams(
        state="st",
        scopes=None,
        code_challenge="challenge",
        redirect_uri=AnyUrl(REDIRECT),
        redirect_uri_provided_explicitly=True,
        resource=resource,
    )


async def _pending(db, url: str) -> OAuthPendingAuthorization:
    request_id = parse_qs(urlparse(url).query)["request"][0]
    return await db.get(OAuthPendingAuthorization, request_id)


async def _code(provider, db) -> str:
    url = await provider.authorize(CLIENT, _params())
    pending = await _pending(db, url)
    redirect = await provider.approve(pending.id, pending.csrf, USER)
    q = parse_qs(urlparse(redirect).query)
    assert q["state"] == ["st"]
    return q["code"][0]


async def _tokens(provider, db):
    code = await provider.load_authorization_code(CLIENT, await _code(provider, db))
    return await provider.exchange_authorization_code(CLIENT, code)


@pytest.mark.asyncio
async def test_client_roundtrip(provider):
    got = await provider.get_client("cid")
    assert got is not None and got.client_id == "cid" and got.redirect_uris == CLIENT.redirect_uris
    assert await provider.get_client("other") is None


@pytest.mark.asyncio
async def test_authorize_requires_registered_resource(provider, db):
    for resource in (None, "https://other.example.org/mcp"):
        with pytest.raises(AuthorizeError) as exc:
            await provider.authorize(CLIENT, _params(resource))
        assert exc.value.error == "invalid_request"


@pytest.mark.asyncio
async def test_authorize_rejects_revoked_resource(provider, db):
    res = await store.find_mcp_resource_by_url(db, RES)
    res.revoked_at = utcnow()
    db.add(res)
    await db.commit()
    with pytest.raises(AuthorizeError):
        await provider.authorize(CLIENT, _params())


@pytest.mark.asyncio
async def test_authorize_returns_consent_url_and_stores_pending(provider, db):
    url = await provider.authorize(CLIENT, _params())
    assert url.startswith(f"{ISSUER}/consent?request=")
    pending = await _pending(db, url)
    assert pending is not None and pending.client_id == "cid" and pending.csrf
    assert pending.params["code_challenge"] == "challenge"
    assert pending.params["resource"] == RES
    ttl = pending.expires_at - utcnow()
    assert timedelta(seconds=595) < ttl <= timedelta(seconds=600)


@pytest.mark.asyncio
async def test_approve_rejects_bad_csrf_and_expired_request(provider, db):
    url = await provider.authorize(CLIENT, _params())
    pending = await _pending(db, url)
    pid, csrf = pending.id, pending.csrf
    with pytest.raises(AuthorizeError):
        await provider.approve(pid, "wrong", USER)
    pending.expires_at = utcnow() - timedelta(seconds=1)
    db.add(pending)
    await db.commit()
    with pytest.raises(AuthorizeError):
        await provider.approve(pid, csrf, USER)
    db.expunge_all()
    assert await db.get(OAuthPendingAuthorization, pid) is None  # expired row cleaned up
    assert (await db.exec(select(OAuthGrant))).all() == []


@pytest.mark.asyncio
async def test_approve_creates_grant_and_code_and_consumes_pending(provider, db):
    url = await provider.authorize(CLIENT, _params())
    pending = await _pending(db, url)
    pid, csrf = pending.id, pending.csrf
    redirect = await provider.approve(pid, csrf, USER)
    assert redirect.startswith(REDIRECT + "?")
    db.expunge_all()
    assert await db.get(OAuthPendingAuthorization, pid) is None
    code_plain = parse_qs(urlparse(redirect).query)["code"][0]
    row = await db.get(OAuthCode, store.hash_secret(code_plain))
    assert row is not None and row.code_challenge == "challenge" and row.resource == RES
    assert timedelta(seconds=55) < row.expires_at - utcnow() <= timedelta(seconds=60)
    grant = await db.get(OAuthGrant, row.grant_id)
    assert (grant.user_sub, grant.roles, grant.resource) == ("u1", ["r1", "r2"], RES)
    with pytest.raises(AuthorizeError):  # pending row is single use
        await provider.approve(pid, csrf, USER)


@pytest.mark.asyncio
async def test_code_is_single_use(provider, db):
    plain = await _code(provider, db)
    code = await provider.load_authorization_code(CLIENT, plain)
    assert code.code_challenge == "challenge" and code.subject == "u1" and code.resource == RES
    assert str(code.redirect_uri) == REDIRECT and code.redirect_uri_provided_explicitly
    tokens = await provider.exchange_authorization_code(CLIENT, code)
    assert tokens.token_type == "Bearer" and tokens.expires_in == 3600 and tokens.refresh_token
    assert await provider.load_authorization_code(CLIENT, plain) is None
    with pytest.raises(TokenError) as exc:
        await provider.exchange_authorization_code(CLIENT, code)
    assert exc.value.error == "invalid_grant"


@pytest.mark.asyncio
async def test_code_expires_after_60s(provider, db):
    plain = await _code(provider, db)
    row = await db.get(OAuthCode, store.hash_secret(plain))
    row.expires_at = utcnow() - timedelta(seconds=1)
    db.add(row)
    await db.commit()
    assert await provider.load_authorization_code(CLIENT, plain) is None


@pytest.mark.asyncio
async def test_access_token_audience_and_subject(provider, db):
    tokens = await _tokens(provider, db)
    at = await provider.load_access_token(tokens.access_token)
    assert (at.client_id, at.scopes, at.resource, at.subject) == ("cid", [], RES, "u1")
    row = await db.get(OAuthToken, store.hash_secret(tokens.access_token))
    assert row.audience == RES
    assert 3595 < at.expires_at - time.time() <= 3600


@pytest.mark.asyncio
async def test_refresh_rotates_and_keeps_roles(provider, db, monkeypatch):
    async def no_network(*a, **k):
        raise AssertionError("refresh must not call the identity provider")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    tokens = await _tokens(provider, db)
    rt = await provider.load_refresh_token(CLIENT, tokens.refresh_token)
    assert (rt.client_id, rt.resource, rt.subject) == ("cid", RES, "u1")
    new = await provider.exchange_refresh_token(CLIENT, rt, [])
    assert new.access_token != tokens.access_token and new.refresh_token != tokens.refresh_token
    assert new.expires_in == 3600
    assert await provider.load_access_token(new.access_token) is not None
    assert await provider.load_refresh_token(CLIENT, new.refresh_token) is not None
    row = await db.get(OAuthToken, store.hash_secret(new.access_token))
    grant = await db.get(OAuthGrant, row.grant_id)
    assert grant.roles == ["r1", "r2"] and grant.revoked_at is None
    old = await db.get(OAuthToken, store.hash_secret(tokens.refresh_token))
    assert old.used_at is not None and old.grant_id == grant.id


@pytest.mark.asyncio
async def test_refresh_stops_at_grant_expiry(provider, db):
    tokens = await _tokens(provider, db)
    rt = await provider.load_refresh_token(CLIENT, tokens.refresh_token)
    row = await db.get(OAuthToken, store.hash_secret(tokens.refresh_token))
    grant = await db.get(OAuthGrant, row.grant_id)
    assert row.expires_at == grant.expires_at
    grant.expires_at = utcnow() - timedelta(seconds=1)
    db.add(grant)
    await db.commit()
    assert await provider.load_refresh_token(CLIENT, tokens.refresh_token) is None
    with pytest.raises(TokenError) as exc:
        await provider.exchange_refresh_token(CLIENT, rt, [])
    assert exc.value.error == "invalid_grant"


@pytest.mark.asyncio
async def test_reused_refresh_token_revokes_grant(provider, db):
    tokens = await _tokens(provider, db)
    rt = await provider.load_refresh_token(CLIENT, tokens.refresh_token)
    new = await provider.exchange_refresh_token(CLIENT, rt, [])
    # Replay of the rotated token (raced past load) → grant revoked.
    with pytest.raises(TokenError) as exc:
        await provider.exchange_refresh_token(CLIENT, rt, [])
    assert exc.value.error == "invalid_grant"
    assert await provider.load_access_token(new.access_token) is None
    assert await provider.load_refresh_token(CLIENT, new.refresh_token) is None


@pytest.mark.asyncio
async def test_loading_used_refresh_token_revokes_grant(provider, db):
    tokens = await _tokens(provider, db)
    rt = await provider.load_refresh_token(CLIENT, tokens.refresh_token)
    new = await provider.exchange_refresh_token(CLIENT, rt, [])
    assert await provider.load_refresh_token(CLIENT, tokens.refresh_token) is None
    assert await provider.load_access_token(new.access_token) is None


@pytest.mark.asyncio
async def test_load_access_token_rejects_refresh_and_api_tokens(provider, db):
    tokens = await _tokens(provider, db)
    assert await provider.load_access_token(tokens.refresh_token) is None
    row = await db.get(OAuthToken, store.hash_secret(tokens.access_token))
    grant = await db.get(OAuthGrant, row.grant_id)
    api_plain, _ = await store.issue_token(db, grant=grant, kind="api", audience=RES)
    assert await provider.load_access_token(api_plain) is None
    assert await provider.load_access_token("nope") is None


@pytest.mark.asyncio
async def test_revoke_token_revokes_whole_grant(provider, db):
    tokens = await _tokens(provider, db)
    at = await provider.load_access_token(tokens.access_token)
    await provider.revoke_token(at)
    assert await provider.load_access_token(tokens.access_token) is None
    assert await provider.load_refresh_token(CLIENT, tokens.refresh_token) is None
