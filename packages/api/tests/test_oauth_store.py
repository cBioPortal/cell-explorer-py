"""Token and key store for the MCP OAuth server."""

from datetime import timedelta

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.db.models import utcnow
from cell_explorer_api.services import oauth_store as store

RES = "https://mcp.example.org/mcp"
USER = User(sub="u1", name="A", email="a@b.c", roles=["r1"])


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with AsyncSession(engine) as s:
            yield s
    finally:
        await engine.dispose()


async def _mark_used(db, tok):
    tok.used_at = utcnow()
    db.add(tok)
    await db.commit()


async def _grant(db):
    return await store.create_grant(db, user=USER, client_id="cid", resource=RES)


@pytest.mark.asyncio
async def test_create_grant_copies_user_and_expires_in_30_days(db):
    g = await _grant(db)
    assert (g.user_sub, g.user_email, g.user_name, g.roles) == ("u1", "a@b.c", "A", ["r1"])
    assert g.client_id == "cid" and g.resource == RES
    assert timedelta(days=29, hours=23) < g.expires_at - utcnow() <= timedelta(days=30)


@pytest.mark.asyncio
async def test_tokens_stored_hashed(db):
    g = await _grant(db)
    plain, tok = await store.issue_token(db, grant=g, kind="access", audience=RES)
    assert tok.token_hash == store.hash_secret(plain) != plain
    rows = (await db.exec(text("SELECT * FROM oauth_tokens"))).all()
    assert len(rows) == 1
    assert plain not in [str(v) for v in rows[0]]


@pytest.mark.asyncio
async def test_find_token_respects_kind_and_expiry(db):
    g = await _grant(db)
    plain, tok = await store.issue_token(db, grant=g, kind="access", audience=RES)
    assert (await store.find_token(db, plain, kind="access"))[0].token_hash == tok.token_hash
    assert await store.find_token(db, plain, kind="api") is None
    assert await store.find_token(db, "nope", kind="access") is None
    tok.expires_at = utcnow() - timedelta(seconds=1)
    db.add(tok)
    await db.commit()
    assert await store.find_token(db, plain, kind="access") is None


@pytest.mark.asyncio
async def test_token_lifetimes(db):
    g = await _grant(db)
    for kind, ttl in (("access", 3600), ("api", 600)):
        _, t = await store.issue_token(db, grant=g, kind=kind, audience=RES)
        assert timedelta(seconds=ttl - 5) < t.expires_at - utcnow() <= timedelta(seconds=ttl)
    _, r = await store.issue_token(db, grant=g, kind="refresh", audience=RES)
    assert r.expires_at == g.expires_at


@pytest.mark.asyncio
async def test_token_expiry_capped_by_grant(db):
    g = await _grant(db)
    g.expires_at = utcnow() + timedelta(seconds=10)
    db.add(g)
    await db.commit()
    await db.refresh(g)
    _, t = await store.issue_token(db, grant=g, kind="access", audience=RES)
    assert t.expires_at <= g.expires_at


@pytest.mark.asyncio
async def test_revoked_grant_invalidates_its_tokens(db):
    g = await _grant(db)
    plain, _ = await store.issue_token(db, grant=g, kind="access", audience=RES)
    await store.revoke_grant(db, g.id)
    assert await store.find_token(db, plain, kind="access") is None


@pytest.mark.asyncio
async def test_expired_grant_invalidates_its_tokens(db):
    g = await _grant(db)
    plain, _ = await store.issue_token(db, grant=g, kind="access", audience=RES)
    g.expires_at = utcnow() - timedelta(seconds=1)
    db.add(g)
    await db.commit()
    assert await store.find_token(db, plain, kind="access") is None


@pytest.mark.asyncio
async def test_used_refresh_token_not_found(db):
    g = await _grant(db)
    plain, tok = await store.issue_token(db, grant=g, kind="refresh", audience=RES)
    assert await store.find_token(db, plain, kind="refresh") is not None
    await _mark_used(db, tok)
    assert await store.find_token(db, plain, kind="refresh") is None


@pytest.mark.asyncio
async def test_find_token_allow_used_returns_rotated_refresh_token(db):
    g = await _grant(db)
    plain, tok = await store.issue_token(db, grant=g, kind="refresh", audience=RES)
    await _mark_used(db, tok)
    found = await store.find_token(db, plain, kind="refresh", allow_used=True)
    assert found is not None and found[0].used_at is not None
    await store.revoke_grant(db, g.id)
    assert await store.find_token(db, plain, kind="refresh", allow_used=True) is None


@pytest.mark.asyncio
async def test_revoke_user_grants_counts_and_only_that_user(db):
    g1 = await _grant(db)
    await _grant(db)
    other = await store.create_grant(
        db, user=User(sub="u2", name="B", email="b@b.c"), client_id="cid", resource=RES
    )
    p1, _ = await store.issue_token(db, grant=g1, kind="access", audience=RES)
    p3, _ = await store.issue_token(db, grant=other, kind="access", audience=RES)
    assert await store.revoke_user_grants(db, "u1") == 2
    assert await store.revoke_user_grants(db, "u1") == 0
    assert await store.find_token(db, p1, kind="access") is None
    assert await store.find_token(db, p3, kind="access") is not None


@pytest.mark.asyncio
async def test_mcp_key_shown_once_and_found_by_hash(db):
    key, res = await store.create_mcp_resource(db, url=RES, name="x")
    assert res.key_hash == store.hash_secret(key) != key
    rows = (await db.exec(text("SELECT * FROM mcp_resources"))).all()
    assert key not in [str(v) for v in rows[0]]
    assert (await store.find_mcp_resource_by_key(db, key)).id == res.id
    assert await store.find_mcp_resource_by_key(db, "wrong") is None
    assert (await store.find_mcp_resource_by_url(db, RES)).id == res.id
    assert await store.find_mcp_resource_by_url(db, "https://other") is None


@pytest.mark.asyncio
async def test_revoked_mcp_resource_key_not_found(db):
    key, res = await store.create_mcp_resource(db, url=RES, name="x")
    res.revoked_at = utcnow()
    db.add(res)
    await db.commit()
    assert await store.find_mcp_resource_by_key(db, key) is None
