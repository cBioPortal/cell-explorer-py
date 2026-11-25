"""OAuth authorization-server provider for MCP sign-in, driven directly."""

import time
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import pytest_asyncio
from mcp.server.auth.provider import AuthorizationParams, AuthorizeError, TokenError
from mcp.shared.auth import InvalidRedirectUriError, OAuthClientInformationFull
from pydantic import AnyUrl
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth import oauth_provider
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


def _params(resource: str | None = RES, scopes: list[str] | None = None) -> AuthorizationParams:
    return AuthorizationParams(
        state="st",
        scopes=scopes,
        code_challenge="challenge",
        redirect_uri=AnyUrl(REDIRECT),
        redirect_uri_provided_explicitly=True,
        resource=resource,
    )


async def _pending(db, url: str) -> OAuthPendingAuthorization:
    request_id = parse_qs(urlparse(url).query)["request"][0]
    return await db.get(OAuthPendingAuthorization, request_id)


async def _code(provider, db, scopes: list[str] | None = None) -> str:
    url = await provider.authorize(CLIENT, _params(scopes=scopes))
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


@pytest.mark.asyncio
async def test_requested_scopes_survive_exchange_and_refresh(provider, db):
    code = await provider.load_authorization_code(CLIENT, await _code(provider, db, ["read", "write"]))
    assert code.scopes == ["read", "write"]
    tokens = await provider.exchange_authorization_code(CLIENT, code)
    assert tokens.scope == "read write"
    assert (await provider.load_access_token(tokens.access_token)).scopes == ["read", "write"]
    rt = await provider.load_refresh_token(CLIENT, tokens.refresh_token)
    requested = "read write".split(" ")
    # The SDK TokenHandler refuses any requested scope not on the refresh token.
    assert all(s in rt.scopes for s in requested)
    new = await provider.exchange_refresh_token(CLIENT, rt, requested)
    assert new.scope == "read write"
    assert (await provider.load_access_token(new.access_token)).scopes == ["read", "write"]


@pytest.mark.asyncio
async def test_no_scopes_requested_gives_no_scope(provider, db):
    tokens = await _tokens(provider, db)
    assert tokens.scope is None
    assert (await provider.load_access_token(tokens.access_token)).scopes == []


@pytest.mark.asyncio
async def test_expires_in_comes_from_access_token_capped_by_grant(provider, db):
    tokens = await _tokens(provider, db)
    row = await db.get(OAuthToken, store.hash_secret(tokens.access_token))
    grant = await db.get(OAuthGrant, row.grant_id)
    grant.expires_at = utcnow() + timedelta(seconds=10)
    db.add(grant)
    await db.commit()
    rt = await provider.load_refresh_token(CLIENT, tokens.refresh_token)
    new = await provider.exchange_refresh_token(CLIENT, rt, [])
    assert 1 <= new.expires_in <= 10


@pytest.mark.asyncio
async def test_expires_in_never_below_one(provider, db, monkeypatch):
    code = await provider.load_authorization_code(CLIENT, await _code(provider, db))
    # The access token row already expired by the time expires_in is computed.
    later = utcnow() + timedelta(hours=2)
    monkeypatch.setattr(oauth_provider, "utcnow", lambda: later)
    tokens = await provider.exchange_authorization_code(CLIENT, code)
    assert tokens.expires_in == 1


# Loopback redirect URIs (RFC 8252 7.3)


@pytest.mark.asyncio
async def test_loopback_redirect_any_port_from_stored_client(provider):
    await provider.register_client(
        OAuthClientInformationFull(
            client_id="native",
            redirect_uris=[AnyUrl("http://127.0.0.1:1234/cb"), AnyUrl("http://[::1]/cb"), AnyUrl("https://app.test/cb")],
            token_endpoint_auth_method="none",
        )
    )
    client = await provider.get_client("native")
    for ok in ("http://127.0.0.1:1234/cb", "http://127.0.0.1:9/cb", "http://[::1]:5678/cb", "https://app.test/cb"):
        assert str(client.validate_redirect_uri(AnyUrl(ok))) == ok
    for bad in (
        "http://127.0.0.1:9/cb2",
        "http://localhost:9/cb",
        "https://127.0.0.1:9/cb",
        "https://app.test:8443/cb",
        "http://127.0.0.1.evil.test:9/cb",
    ):
        with pytest.raises(InvalidRedirectUriError):
            client.validate_redirect_uri(AnyUrl(bad))


# Cleanup of expired and abandoned rows


def _ago(seconds: float):
    return utcnow() - timedelta(seconds=seconds)


async def _grant(db, *, client_id="cid", **fields) -> OAuthGrant:
    grant = await store.create_grant(db, user=USER, client_id=client_id, resource=RES)
    for name, value in fields.items():
        setattr(grant, name, value)
    db.add(grant)
    await db.commit()
    await db.refresh(grant)
    return grant


def _pending_row(pid: str, expires_at, client_id="cid") -> OAuthPendingAuthorization:
    return OAuthPendingAuthorization(id=pid, client_id=client_id, params={}, csrf="c", expires_at=expires_at)


def _code_row(code_hash: str, grant: OAuthGrant, expires_at) -> OAuthCode:
    return OAuthCode(
        code_hash=code_hash,
        grant_id=grant.id,
        client_id="cid",
        code_challenge="x",
        redirect_uri=REDIRECT,
        redirect_uri_provided_explicitly=True,
        resource=RES,
        expires_at=expires_at,
    )


async def _ids(db, model, key: str) -> set:
    db.expunge_all()
    return {getattr(r, key) for r in (await db.exec(select(model))).all()}


@pytest.mark.asyncio
async def test_purge_deletes_expired_pending_authorizations(db):
    db.add(_pending_row("old", _ago(1)))
    db.add(_pending_row("live", utcnow() + timedelta(seconds=60)))
    await db.commit()
    await store.purge_expired(db)
    assert await _ids(db, OAuthPendingAuthorization, "id") == {"live"}


@pytest.mark.asyncio
async def test_purge_deletes_expired_codes(db):
    grant = await _grant(db)
    await store.issue_token(db, grant=grant, kind="refresh", audience=RES)  # keeps the grant
    db.add(_code_row("old", grant, _ago(1)))
    db.add(_code_row("live", grant, utcnow() + timedelta(seconds=60)))
    await db.commit()
    await store.purge_expired(db)
    assert await _ids(db, OAuthCode, "code_hash") == {"live"}


@pytest.mark.asyncio
async def test_purge_deletes_expired_tokens(db):
    grant = await _grant(db)
    _, old = await store.issue_token(db, grant=grant, kind="access", audience=RES)
    old_hash = old.token_hash
    old.expires_at = _ago(1)
    db.add(old)
    await db.commit()
    _, live = await store.issue_token(db, grant=grant, kind="refresh", audience=RES)
    live_hash = live.token_hash
    await store.purge_expired(db)
    assert await _ids(db, OAuthToken, "token_hash") == {live_hash}
    assert old_hash != live_hash


@pytest.mark.asyncio
async def test_purge_deletes_expired_and_revoked_grants_with_their_rows(db):
    live = await _grant(db)
    live_id = live.id
    expired = await _grant(db, expires_at=_ago(1))
    revoked = await _grant(db, revoked_at=_ago(1))
    for grant in (live, expired, revoked):
        await store.issue_token(db, grant=grant, kind="refresh", audience=RES)
        db.add(_code_row(f"code-{grant.id}", grant, utcnow() + timedelta(seconds=60)))
        await db.commit()
    await store.purge_expired(db)
    assert await _ids(db, OAuthGrant, "id") == {live_id}
    assert await _ids(db, OAuthToken, "grant_id") == {live_id}
    assert await _ids(db, OAuthCode, "grant_id") == {live_id}


@pytest.mark.asyncio
async def test_purge_deletes_grants_never_exchanged_after_10_minutes(db):
    fresh = await _grant(db, created_at=_ago(9 * 60))
    keep = {fresh.id}
    abandoned = await _grant(db, created_at=_ago(11 * 60))
    db.add(_code_row("abandoned", abandoned, _ago(10 * 60)))
    await db.commit()
    used = await _grant(db, created_at=_ago(11 * 60))
    keep.add(used.id)
    await store.issue_token(db, grant=used, kind="refresh", audience=RES)
    await store.purge_expired(db)
    assert await _ids(db, OAuthGrant, "id") == keep
    assert await _ids(db, OAuthCode, "code_hash") == set()


@pytest.mark.asyncio
async def test_authorize_cleans_up_at_most_once_a_minute(provider, db):
    db.add(_pending_row("old-1", _ago(1)))
    await db.commit()
    await provider.authorize(CLIENT, _params())
    assert "old-1" not in await _ids(db, OAuthPendingAuthorization, "id")

    db.add(_pending_row("old-2", _ago(1)))
    await db.commit()
    await provider.authorize(CLIENT, _params())
    assert "old-2" in await _ids(db, OAuthPendingAuthorization, "id")  # throttled

    provider._next_cleanup = 0.0  # a minute later
    await provider.authorize(CLIENT, _params())
    assert "old-2" not in await _ids(db, OAuthPendingAuthorization, "id")


@pytest.mark.asyncio
async def test_code_exchange_cleans_up(provider, db):
    code = await provider.load_authorization_code(CLIENT, await _code(provider, db))
    db.add(_pending_row("old", _ago(1)))
    await db.commit()
    provider._next_cleanup = 0.0
    await provider.exchange_authorization_code(CLIENT, code)
    assert "old" not in await _ids(db, OAuthPendingAuthorization, "id")


@pytest.mark.asyncio
async def test_pending_authorizations_capped_per_client(provider, db):
    other = OAuthClientInformationFull(
        client_id="other", redirect_uris=[AnyUrl(REDIRECT)], token_endpoint_auth_method="none"
    )
    await provider.register_client(other)
    for i in range(5):
        db.add(_pending_row(f"expired-{i}", _ago(1)))  # expired rows never count
    await db.commit()
    provider._next_cleanup = float("inf")  # keep the expired rows in place
    for _ in range(oauth_provider.MAX_PENDING_PER_CLIENT):
        await provider.authorize(CLIENT, _params())
    with pytest.raises(AuthorizeError) as exc:
        await provider.authorize(CLIENT, _params())
    assert exc.value.error == "temporarily_unavailable"
    assert oauth_provider.MAX_PENDING_PER_CLIENT == 50
    await provider.authorize(other, _params())  # other clients unaffected
